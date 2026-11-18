"""
실기 조종 시 데이터셋 녹화 — control_keyboard.py / control_gamepad.py 에서 공통 사용.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import cv2

DEFAULT_PERSON_NAME = "unknown_person"
DEFAULT_TASK_NAME = "Pick and Place"

DATASET_ROOT = os.environ.get("CAPSTONE_DATASET_ROOT", "dataset")
RECORD_FPS = 10
# Dynamixel 관절 ID (control_* 스크립트의 DXL_IDs 와 동일 순서)
JOINT_IDS = (11, 22, 23, 24, 25)
JOINTS_CSV_NAME = "joints.csv"
METADATA_CSV_NAME = "metadata.csv"


def _clean_path_part(value: str, fallback: str) -> str:
    value = (value or "").strip()
    if not value:
        value = fallback
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"\s+", "_", value)
    return value or fallback


def _dataset_names() -> tuple[str, str, str]:
    meta = os.environ.get("CAPSTONE_DATASET_META")
    if meta:
        parts = [part.strip() for part in re.split(r"[,/|]", meta)]
        if len(parts) >= 3:
            return (
                _clean_path_part(parts[0], DEFAULT_PERSON_NAME),
                _clean_path_part(parts[1], DEFAULT_TASK_NAME),
            )
        print("⚠️ CAPSTONE_DATASET_META는 person,task형식이어야 합니다. 개별 환경변수를 사용합니다.")

    return (
        _clean_path_part(os.environ.get("CAPSTONE_PERSON_NAME", ""), DEFAULT_PERSON_NAME),
        _clean_path_part(os.environ.get("CAPSTONE_TASK_NAME", ""), DEFAULT_TASK_NAME),
    )


class DatasetRecorder:
    """X 토글로 세션 시작/종료. 세션당 10Hz 로 image_0~2 PNG + joints.csv 행 적재."""

    __slots__ = (
        "_recording",
        "_session_dir",
        "_joints_file",
        "_joints_writer",
        "_last_timestep",
        "_last_sample_perf",
        "_interval",
        "_person_name",
        "_task_name",
        "_last_session_dir",
    )

    def __init__(self) -> None:
        self._recording = False
        self._session_dir: Path | None = None
        self._joints_file: TextIO | None = None
        self._joints_writer: Any = None
        self._last_timestep = 0
        self._last_sample_perf = 0.0
        self._interval = 1.0 / float(RECORD_FPS)
        self._person_name, self._task_name = _dataset_names()
        self._last_session_dir: Path | None = None
        print(
            "💾 저장 경로 설정: "
            f"{Path(DATASET_ROOT) / self._person_name / self._task_name}/<timestamp>"
        )

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def last_timestep(self) -> int:
        """마지막으로 디스크에 쓴 timestep (0이면 아직 한 프레임도 없음)."""
        return self._last_timestep

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def toggle_recording(self) -> bool:
        """
        녹화 중이면 중지하고 False, 아니면 새 세션을 열고 True.
        """
        if self._recording:
            self._close_session()
            self._recording = False
            print("⏹ 녹화 중지.")
            return False

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = (
            Path(DATASET_ROOT)
            / self._person_name
            / self._task_name
            / stamp
        )
        root.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (root / f"image_{i}").mkdir(exist_ok=True)
        self._write_metadata(root, stamp)

        joints_path = root / JOINTS_CSV_NAME
        f = open(joints_path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["timestep", "t_seconds", "id11", "id22", "id23", "id24", "id25"])

        self._session_dir = root
        self._joints_file = f
        self._joints_writer = w
        self._last_timestep = 0
        # 첫 유효 샘플을 곧바로 한 번 받을 수 있도록 간격 리셋
        self._last_sample_perf = time.perf_counter() - self._interval
        self._recording = True
        print(f"🔴 녹화 시작 → {root}")
        return True

    def delete_last_recording(self) -> bool:
        """
        현재 녹화 중이면 해당 세션을 닫고 삭제한다.
        녹화 중이 아니면 마지막으로 닫힌 세션, 없으면 디스크의 최신 세션을 삭제한다.
        """
        target = self._session_dir if self._recording else self._last_session_dir
        if self._recording:
            self._close_session()
            self._recording = False

        if target is None or not target.exists():
            target = self._find_latest_session_dir()

        if target is None or not target.exists():
            print("\n🗑 삭제할 녹화 세션이 없습니다.")
            return False

        try:
            shutil.rmtree(target)
        except Exception as exc:
            print(f"\n⚠️ 녹화 세션 삭제 실패: {target} ({exc})")
            return False

        if self._last_session_dir == target:
            self._last_session_dir = None
        print(f"\n🗑 마지막 녹화 세션 삭제: {target}")
        return True

    def _write_metadata(self, root: Path, stamp: str) -> None:
        metadata_path = root / METADATA_CSV_NAME
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["person_name", self._person_name])
            writer.writerow(["task_name", self._task_name])
            writer.writerow(["started_at", stamp])
            writer.writerow(["record_fps", RECORD_FPS])

    def maybe_record_frame(
        self,
        frames: tuple[Any, Any, Any],
        positions: dict[int, int],
    ) -> None:
        """
        녹화 중이고 RECORD_FPS 간격이 지났으며 관절 통신이 전부 유효할 때만
        timestep 을 1 증가시키고 PNG + joints.csv 1행을 쓴다.
        관절이 하나라도 -1 이면 해당 간격은 통째로 건너뛴다(행·이미지 없음).
        """
        if not self._recording or self._session_dir is None or self._joints_writer is None:
            return
        if not all(positions.get(i, -1) != -1 for i in JOINT_IDS):
            return

        now = time.perf_counter()
        if now - self._last_sample_perf < self._interval:
            return

        self._last_timestep += 1
        t = self._last_timestep
        self._last_sample_perf = now
        name = f"{t:04d}.png"

        for idx, frame in enumerate(frames):
            if frame is not None:
                out = self._session_dir / f"image_{idx}" / name
                cv2.imwrite(str(out), frame)

        t_sec = (t - 1) / float(RECORD_FPS)
        row = [t, t_sec] + [int(positions[i]) for i in JOINT_IDS]
        self._joints_writer.writerow(row)
        self._joints_file.flush()

    def finalize(self) -> None:
        """프로그램 종료 시 녹화 중이면 세션을 닫는다."""
        if self._recording:
            self._close_session()
            self._recording = False
            print("⏹ 종료에 따라 녹화 세션을 닫았습니다.")

    def _close_session(self) -> None:
        closed_dir = self._session_dir
        if self._joints_file is not None:
            try:
                self._joints_file.close()
            except Exception:
                pass
            self._joints_file = None
            self._joints_writer = None
        self._session_dir = None
        if closed_dir is not None:
            self._last_session_dir = closed_dir

    def _find_latest_session_dir(self) -> Path | None:
        base = Path(DATASET_ROOT) / self._person_name / self._task_name
        if not base.exists():
            return None
        candidates = [path for path in base.iterdir() if path.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)
