#!/usr/bin/env bash
# ============================================================================
# Robot-PC side. Pairs with serve.sh on the GPU machine.
#
#   ./client.sh "push the knife to the right"          한 번 굴린다
#   ./client.sh "push the knife to the right" --dry-run   모터는 안 건드린다
#
# 첫 인자가 지시문이고, 나머지는 그대로 넘어간다(뒤에 온 것이 이긴다).
#
# 이 파일과 smolvla_client_local.py 둘만 로봇 PC에 있으면 된다.
# 서버 IP는 아래 SERVER를 고칠 것.
# ============================================================================
set -eu
cd "$(dirname "$0")"

# !!! 여기를 GPU 머신의 실제 IP로 고쳐야 한다. 서버는 0.0.0.0에 바인딩하지만
# 방화벽이 막고 있으면 GPU 머신에서 한 번만: sudo ufw allow 8000/tcp
SERVER=${SERVER:-http://<GPU_SERVER_IP>:8000}   # 실제 IP로 교체하거나 SERVER=http://x.x.x.x:8000 ./client.sh ...
LANG_TEXT="${1:?usage: ./client.sh \"<instruction>\" [extra flags]}"; shift || true
OUT=runs/rollout_$(date +%m%d_%H%M%S)

# 처음 한 번은 카메라부터 확인한다. 두 대의 C270이 같은 시리얼을 쓰기 때문에
# by-id 심볼릭 링크가 하나만 생기고, 순서가 뒤바뀌면 조용히 엉뚱하게 동작한다.
#   python smolvla_client_local.py --list-cameras
# 기대 배치(사용자가 로봇을 바라보는 기준):
#   camera1 로봇 오른쪽 위에서 대각 아래 / camera2 정면 하단 / camera3 로봇 왼쪽 위에서 대각 아래

python smolvla_client_local.py \
    --url "$SERVER" \
    --lang "$LANG_TEXT" \
    --out-dir "$OUT" \
    \
    `# --- 로봇 -------------------------------------------------------------` \
    --port /dev/ttyUSB0 \
    `# --baud 1000000` \
    `# --dxl-ids 11,22,23,24,25` \
    `# --arm-delay-s 0.0                 첫 명령 전 대기` \
    \
    `# --- 제어 주기 --------------------------------------------------------` \
    `# 서버가 주는 청크에서 몇 스텝을 실행할지. 서버의 --emit-steps와 맞춘다.` \
    --exec-steps 8 \
    `# 한 스텝의 길이. 데이터가 10Hz라 100ms가 정답이고, 바꾸면 정책의 틱 변화량이` \
    `# 실제 속도와 어긋난다.` \
    --step-dt-ms 100 \
    `# --ramp-steps 5                    첫 청크로 갈 때 몇 스텝에 걸쳐 접근할지` \
    `# --max-chunks 0                    0 = 무제한` \
    \
    `# --- 떨림 억제 (클라이언트 쪽) ----------------------------------------` \
    `# 시간축 가우시안. 청크 16 중 8만 실행하므로 미래가 이미 예측돼 있어 지연이 0이다.` \
    `# 커널 7 / 시그마 1.0이 측정상 최적이고, 기본값(3 / 0.8)은 더 약하다.` \
    --smooth-actions --smooth-kernel 7 --smooth-sigma 1.0 \
    `# --smooth-gripper                  파지 채널까지 뭉갠다. 기본은 제외.` \
    \
    `# --- 안전 -------------------------------------------------------------` \
    `# 한 스텝에 허용할 최대 이동. 떨림을 가리기만 하고 없애지는 않는다.` \
    `# 150은 클라이언트 기본값이자 지금까지 쓰던 값이다. 낮추면 팔이 정책보다 뒤처진다 --` \
    `# 정책의 스텝 크기가 실측 87틱까지 나오므로 60 같은 값을 주면 상시 클리핑된다.` \
    --max-step-ticks 150 \
    `# --joint-limits ...                모터별 하한/상한` \
    `# --dry-run                         모터 쓰기 없이 전 구간을 돌린다` \
    \
    `# --- 기록 -------------------------------------------------------------` \
    `# off | all | flagged. flagged = 클리핑이나 rate limit이 걸린 스텝만 저장.` \
    --save-frames flagged \
    `# --jpeg-quality 85` \
    `# --label                           끝나고 성공/실패를 물어본다` \
    "$@"

# ============================================================================
# 처음 붙일 때 순서
#   1. python smolvla_client_local.py --offline-self-test    서버·로봇 없이 안전 계산만
#   2. python smolvla_client_local.py --list-cameras         카메라 3대 매핑 확인
#   3. ./client.sh "..." --dry-run                           모터 안 움직이고 전 구간
#   4. ./client.sh "..."                                     실제 구동
#
# 서버 로그에서 볼 것
#   rtt=..ms slack=+..ms   slack이 음수면 팔이 계획을 다 쓰고 기다린 것이다.
#                          --exec-steps를 더 줄이면 안 된다는 뜻.
#   LATE: ...              같은 뜻의 경고.
#   끝내면 왕복 시간 중앙/p90/최대와 slack 요약이 출력된다.
# ============================================================================
