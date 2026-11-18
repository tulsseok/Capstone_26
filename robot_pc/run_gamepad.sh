#!/usr/bin/env bash
# control_gamepad.py 런처 — image_0 ↔ image_1 카메라 스왑 토글.
#
#   ./run_gamepad.sh            # 스왑 OFF (기본 = 원래대로)
#   ./run_gamepad.sh --swap     # 스왑 ON  (image_0 ↔ image_1)
#   SWAP01=1 ./run_gamepad.sh   # 환경변수로도 ON
#
# 기존 파일은 수정하지 않는다. 자동 선택 결과를 읽어
# CAPSTONE_CAMERA_ORDER 만 계산해 주입한다.
set -euo pipefail

PY=/home/minjun/anaconda3/envs/wildgs-slam/bin/python
cd "$(dirname "$0")"

SWAP="${SWAP01:-0}"
ARGS=()
for a in "$@"; do
  case "$a" in
    --swap)    SWAP=1 ;;
    --no-swap) SWAP=0 ;;
    *)         ARGS+=("$a") ;;
  esac
done

# 이미 CAPSTONE_CAMERA_ORDER 가 있으면 그것을 기준으로, 없으면 자동 선택 결과를 기준으로.
ORDER=$(SWAP="$SWAP" "$PY" - <<'PYEOF'
import os
from src_control.camera_setup import _get_camera_devices, _select_camera_indices

forced = os.environ.get("CAPSTONE_CAMERA_ORDER", "").strip()
if forced:
    idx = [int(x) for x in forced.split(",") if x.strip()]
else:
    idx = _select_camera_indices(_get_camera_devices())

if os.environ.get("SWAP") == "1" and len(idx) >= 2:
    idx[0], idx[1] = idx[1], idx[0]

print(",".join(str(i) for i in idx[:3]))
PYEOF
)

if [ "$SWAP" = "1" ]; then
  echo "🔄 image_0 ↔ image_1 스왑: ON"
else
  echo "➡️  image_0 ↔ image_1 스왑: OFF (원래대로)"
fi
echo "   CAPSTONE_CAMERA_ORDER=$ORDER   (image_0, image_1, image_2 순)"

export CAPSTONE_CAMERA_ORDER="$ORDER"
export DYNAMIXEL_PORT="${DYNAMIXEL_PORT:-/dev/ttyUSB0}"
exec "$PY" control_gamepad.py "${ARGS[@]}"
