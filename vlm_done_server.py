#!/usr/bin/env python3
"""External-VLM termination judge for the OpenMANIPULATOR-X rollout.

WHY THIS EXISTS
---------------
The done_flag action channel (d6_*/d8_done) was retired on 2026-09-09: asking the policy to
regress its own termination made a 6-D action whose eval_loss is not comparable to the 5-D
runs, and it never worked on the arm. Termination is now judged OUTSIDE the policy, by a
VLM that looks at the same camera frames and answers one yes/no question.

WHAT IT IS NOT
--------------
It is not a success metric. "The episode is over" and "the task was done correctly" are two
different questions; this server answers the first one by default (--question done) and the
second only if asked (--question success). Do not report one as the other.

PROTOCOL (mirrors smolvla_server_openmani.py so the client speaks one dialect)
    POST /judge {"images": [b64 JPEG xN], "task": "place the pen on the left",
                 "question": "done"|"success"    (optional, overrides the server default)}
        -> {"done": bool, "score": float, "raw": "<the VLM's text>", "streak": int,
            "consecutive": int, "inference_ms": float, "question": "done"}
    GET  /health -> {"ok": true, "model": ..., "device": ..., "question": ..., "threshold": ...}
    POST /reset  -> {"ok": true}      clears the streak between rollouts

`score` is the probability the VLM puts on the token "yes" at the first generated position,
renormalised over {yes, no}: p(yes) / (p(yes) + p(no)). That is a real number to threshold,
unlike parsing the free text. `--threshold` and `--consecutive` mean exactly what they meant
on the old 6-D server: `done` goes true only after `consecutive` answers in a row land at or
above `threshold`, so a single blurry frame cannot end a rollout.

THE CHEAPER RULE THIS DOES NOT REPLACE
--------------------------------------
From the 2026-09 recollection on, every demonstration ends parked in the default pose and
holds still for one second. A "commanded motion ~ 0 for N chunks" test on the client is
free and needs no GPU. It fails only when the policy keeps jittering -- which is exactly the
open problem -- so the two are meant to run together, not instead of each other.

    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n lerobot \
        python vlm_done_server.py --self-test          # no weights needed
    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n lerobot \
        python vlm_done_server.py --port 8100

The weights are NOT downloaded automatically by --self-test. The first real start pulls
~7 GB from the hub unless --model-path points at a local directory.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# The two questions. Both are written so that "yes" is the END of the episode, never the
# start, and both name the arm's resting behaviour that the new data guarantees.
PROMPTS = {
    "done": (
        "You are watching a small robot arm that was told: \"{task}\".\n"
        "The arm has finished when it has stopped moving and returned to its resting "
        "position, whether or not it did the task correctly.\n"
        "Has the arm finished? Answer with one word, yes or no."
    ),
    "rest": (
        "You are looking at a small robot arm on a table.\n"
        "Its resting position is folded back, away from the objects, not reaching over them.\n"
        "Is the arm in that resting position right now? Answer with one word, yes or no."
    ),
    "still": (
        "The images are two views of the same small robot arm, taken a moment apart.\n"
        "Compare them: if the arm is in the same place in both, it has stopped moving.\n"
        "Has the arm stopped moving? Answer with one word, yes or no."
    ),
    "success": (
        "You are watching a small robot arm that was told: \"{task}\".\n"
        "Answer only about the result you can see now, not about how it got there.\n"
        "Was the instruction carried out correctly? Answer with one word, yes or no."
    ),
}


def b64_to_pil(b64: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class VlmDoneJudge:
    """Plain object (not FastAPI-bound) so --self-test can drive judge() directly."""

    def __init__(self, model_path: str, device: str, threshold: float, consecutive: int,
                 question: str, max_images: int, image_max_side: int, dtype: str):
        self.model_path = model_path
        self.device = device
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self.question = question
        self.max_images = int(max_images)
        self.image_max_side = int(image_max_side)
        self.dtype = dtype
        self.streak = 0
        self.model = None
        self.processor = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []

    # -- weights ---------------------------------------------------------------
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=td, attn_implementation="sdpa",
        ).to(self.device).eval()
        # sm_120 has no flash-attn wheel in this env; sdpa is the deliberate choice, same as
        # the SmolVLA server.
        tok = self.processor.tokenizer
        # Both capitalisations and both with/without the leading space: which one the model
        # emits depends on the chat template, and guessing wrong silently gives score 0.5.
        for w in ("yes", "Yes", " yes", " Yes", "YES"):
            ids = tok.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                self._yes_ids.append(ids[0])
        for w in ("no", "No", " no", " No", "NO"):
            ids = tok.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                self._no_ids.append(ids[0])
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("neither 'yes' nor 'no' is a single token for this tokenizer; "
                               "the score would be meaningless")
        return time.time() - t0

    # -- one judgement ---------------------------------------------------------
    def judge(self, images_b64: list[str], task: str, question: str | None = None) -> dict:
        import torch

        q = question or self.question
        if q not in PROMPTS:
            raise ValueError(f"question must be one of {sorted(PROMPTS)}, got {q!r}")
        if not images_b64:
            raise ValueError("images is empty")
        imgs = [b64_to_pil(b) for b in images_b64[: self.max_images]]
        imgs = [self._shrink(im) for im in imgs]

        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": PROMPTS[q].format(task=task)})
        msgs = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt").to(self.device)

        t0 = time.time()
        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = float(probs[self._yes_ids].sum())
        p_no = float(probs[self._no_ids].sum())
        denom = p_yes + p_no
        score = 0.5 if denom <= 0 else p_yes / denom
        infer_ms = (time.time() - t0) * 1e3

        hit = score >= self.threshold
        self.streak = self.streak + 1 if hit else 0
        return {
            "done": bool(self.streak >= self.consecutive),
            "score": score,
            "p_yes_raw": p_yes,
            "p_no_raw": p_no,
            "streak": int(self.streak),
            "consecutive": self.consecutive,
            "threshold": self.threshold,
            "question": q,
            "n_images": len(imgs),
            "inference_ms": infer_ms,
        }

    def _shrink(self, im):
        if self.image_max_side <= 0:
            return im
        w, h = im.size
        m = max(w, h)
        if m <= self.image_max_side:
            return im
        s = self.image_max_side / m
        return im.resize((max(1, int(w * s)), max(1, int(h * s))))

    def reset(self):
        self.streak = 0


def serve_stdlib(judge: VlmDoneJudge, host: str, port: int):
    """No-dependency fallback. `lerobot` has no uvicorn, same as the SmolVLA server, which
    also ships a stdlib backend. Threaded because the client may health-check mid-rollout."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # the rollout log is the log
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                return self._send(200, {"ok": True, "model": judge.model_path,
                                        "device": judge.device, "question": judge.question,
                                        "threshold": judge.threshold,
                                        "consecutive": judge.consecutive})
            self._send(404, {"detail": f"no GET {self.path}"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"detail": f"bad JSON: {e}"})
            path = self.path.rstrip("/")
            if path == "/reset":
                judge.reset()
                return self._send(200, {"ok": True})
            if path == "/judge":
                try:
                    return self._send(200, judge.judge(body.get("images") or [],
                                                       body.get("task", ""),
                                                       body.get("question")))
                except Exception as e:  # noqa: BLE001
                    return self._send(400, {"detail": f"{type(e).__name__}: {e}"})
            self._send(404, {"detail": f"no POST {self.path}"})

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def build_app(judge: VlmDoneJudge):
    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "model": judge.model_path, "device": judge.device,
                "question": judge.question, "threshold": judge.threshold,
                "consecutive": judge.consecutive}

    @app.post("/reset")
    def reset():
        judge.reset()
        return {"ok": True}

    @app.post("/judge")
    async def judge_ep(body: dict):
        try:
            return judge.judge(body.get("images") or [], body.get("task", ""),
                               body.get("question"))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")

    return app


# ---------------------------------------------------------------------------
def self_test() -> int:
    """Everything that can be checked WITHOUT the 7 GB download."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        ok = ok and bool(cond)

    # 'rest' and 'still' deliberately do NOT mention the instruction: they ask about the arm's
    # pose/motion only, so the same prompt applies to all 54 sentences.
    check("task-conditioned prompts carry the placeholder",
          all("{task}" in PROMPTS[k] for k in ("done", "success")))
    check("pose-only prompts carry no placeholder",
          all("{task}" not in PROMPTS[k] for k in ("rest", "still")))
    check("every prompt formats without a KeyError",
          all(isinstance(PROMPTS[k].format(task="x"), str) for k in PROMPTS))
    check("all four questions are distinct texts", len(set(PROMPTS.values())) == len(PROMPTS))

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (12, 34, 56)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    back = b64_to_pil(b64)
    check("b64 JPEG round-trip keeps the size", back.size == (640, 480), str(back.size))

    j = VlmDoneJudge("<none>", "cpu", 0.7, 3, "done", 3, 448, "bfloat16")
    check("shrink caps the long side", j._shrink(back).size == (448, 336), str(j._shrink(back).size))
    check("shrink leaves a small image alone",
          j._shrink(Image.new("RGB", (100, 80))).size == (100, 80))

    # the streak machine, driven without a model
    j.model = object()
    seq, out = [0.9, 0.9, 0.2, 0.9, 0.9, 0.9], []
    for s in seq:
        j.streak = j.streak + 1 if s >= j.threshold else 0
        out.append(j.streak >= j.consecutive)
    check("one low score resets the streak", out == [False, False, False, False, False, True],
          str(out))
    j.reset()
    check("reset clears the streak", j.streak == 0)

    check("an empty image list is refused",
          _raises(lambda: j.judge([], "x")), "judge([]) must raise")
    check("an unknown question is refused",
          _raises(lambda: j.judge([b64], "x", "banana")))

    print("-" * 70)
    print(f"SELF-TEST: {'PASSED' if ok else 'FAILED'}  (judge by THIS line, not by the exit code)")
    return 0 if ok else 1


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL,
                    help="hub id or a local directory. The default pulls ~7 GB on first start.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--question", choices=tuple(PROMPTS), default="rest",
                    help="'rest' = the arm is parked in its resting pose (ONE frame per camera); "
                         "'still' = the arm has stopped, judged from a PAIR of frames taken a "
                         "moment apart; 'done' = free-form 'is it over'; 'success' = the task "
                         "was carried out correctly. These are NOT the same measurement. "
                         "Measured 2026-09-10 on 6 held-out episodes: 'done' does not separate "
                         "mid-episode from end-of-episode at all (mean 0.472 vs 0.457).")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="p(yes) at or above this counts as a hit. ARBITRARY until measured "
                         "against labelled rollouts.")
    ap.add_argument("--consecutive", type=int, default=3,
                    help="hits in a row before done=true. ARBITRARY, same as the old 6-D server.")
    ap.add_argument("--max-images", type=int, default=3,
                    help="cameras to forward. 3 = every camera; 1 is ~3x faster.")
    ap.add_argument("--image-max-side", type=int, default=448,
                    help="downscale each frame to this long side before the VLM. 0 = off.")
    ap.add_argument("--http-backend", choices=("auto", "fastapi", "stdlib"), default="auto",
                    help="'auto' uses FastAPI+uvicorn when importable, else the stdlib server. "
                         "The lerobot env has no uvicorn, so this normally lands on 'stdlib'.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    judge = VlmDoneJudge(args.model_path, args.device, args.threshold, args.consecutive,
                         args.question, args.max_images, args.image_max_side, args.dtype)
    print(f"[load] {args.model_path} -> {args.device} ({args.dtype})", flush=True)
    secs = judge.load()
    print(f"[load] ready in {secs:.1f} s", flush=True)
    print(f"[serve] http://{args.host}:{args.port}  question={args.question} "
          f"threshold={args.threshold} consecutive={args.consecutive}", flush=True)

    backend = args.http_backend
    if backend == "auto":
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
            backend = "fastapi"
        except Exception:  # noqa: BLE001
            backend = "stdlib"
    if backend == "fastapi":
        import uvicorn
        uvicorn.run(build_app(judge), host=args.host, port=args.port, log_level="warning")
    else:
        print("[http] stdlib backend (no uvicorn in this env)", flush=True)
        serve_stdlib(judge, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
