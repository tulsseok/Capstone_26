#!/usr/bin/env bash
# ============================================================================
# SmolVLA policy server for OpenMANIPULATOR-X.
#
#   ./serve.sh                    the configuration below, as written
#   ./serve.sh --n-avg 8          same, with one flag overridden (later wins)
#
# Every knob is listed. Uncomment a line to turn it on; the value shown is what
# the flag defaults to unless noted. Numbers in comments are measured on this
# repo's own data -- see results/EXPERIMENTS.md for how.
#
# Terms used below, since they are all in ticks (Dynamixel encoder units, 4096
# per revolution):
#   draw 산포   ask the same question 8 times, how far apart the answers are
#   gt 오차     predicted joint values vs what the demonstrator actually did
#   seam        after executing 8 steps and re-predicting, how much the new plan
#               disagrees with the old one over the steps they share
# ============================================================================
set -eu
cd "$(dirname "$0")"

# 로봇 PC가 다른 기기라면 한 번만: sudo ufw allow 8000/tcp
# 서버는 0.0.0.0:8000에 바인딩하므로 방화벽만 열려 있으면 붙는다.

# ---------------------------------------------------------------------------
# WHICH WEIGHTS.  --model-path and --dataset-root must come from the SAME run:
# the dataset supplies the mean/std the ticks are un-normalised with, and the
# 54 instruction strings.  Mixing them silently shifts every output.
# ---------------------------------------------------------------------------
# 아래는 MODEL/DATA 짝 목록이다. 짝을 섞으면 정규화 통계와 문장 목록이 어긋나 출력 틱이
# 통째로 밀린다 -- 한 줄씩 붙어 있는 두 줄을 항상 같이 옮길 것.
#
# 카메라 리그가 두 번 바뀌었다. 지금 리그(0912 이후)와 맞는 것만 실기에 쓸 수 있다.
#   구 리그    : d3 / d4 / d5 / d7 / d8 세대 (0905 이전 배치)
#   현 리그    : d9(0912) / d14(0919) 및 이들의 병합 d15 / d16
#   d10_merged : d8 + d9 -- 두 리그가 섞여 있고 구 리그는 가중치 0.3으로 학습됐다
#
# eval_loss는 홀드아웃이 같을 때만 비교된다. d10_new / d10_mix / d12_* / d15_* 는 전부
# 같은 18개(d9 인덱스 4,15,23,...,237)로 고정해 놨으므로 이 여섯은 서로 비교된다.
# d8 이하 세대는 홀드아웃이 달라 그 숫자와 비교하면 안 된다.

# ================= 모델 고르기 (현 리그, 0912 이후) =================
# 사용법: 아래 목록에서 원하는 줄 맨 앞의 '# ' 하나만 지운다. MODEL 과 DATA 가 한 줄에 짝으로 묶여 있어
# 같이 바뀐다. 뒤에 있는 줄이 앞 줄을 덮어쓰므로 기본값 줄은 그대로 둬도 된다(되돌리려면 다시 '# ').
# 이름: d<번호>_* = 데이터 병합 후 한 번에 학습 / p_* = 0912 위에 단계별(피라미드) / v2_* = 09-30 새 체인.
# eval 은 같은 데이터·같은 홀드아웃끼리만 비교된다. v2_* 는 그리퍼 라벨이 바뀌어 다른 판과 비교 불가.

# --- 현재 기본값 ---
# MODEL=runs/p_place_1e5/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919           # [09-27] 0912 -> 0919 place, lr 1e-5. eval 0.0898

# --- [2026-09-30] v2 체인: 그리퍼 라벨 복원(닫는 중/쥔 동안 2700) + state noise + replay 20% ---
#     lr base 1e-4 / 1단 3e-5 / 2단 1e-5.  _aux = novelty(관절값 못 보는 보조 head 로 다음 잡기/놓기 순간 관절 예측)
# MODEL=runs/v2_base/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # 0912 place 새 base. eval 0.1341
# MODEL=runs/v2_base_aux/checkpoints/004750/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # v2_base + aux. eval 0.1286
# MODEL=runs/v2_place_3e5/checkpoints/003250/pretrained_model; DATA=data/lerobot_v3_d18_place_all          # v2_base -> 0919 place. eval 0.0833
# MODEL=runs/v2_place_3e5_aux/checkpoints/003250/pretrained_model; DATA=data/lerobot_v3_d18_place_all      # 위 + aux. eval 0.0845
# MODEL=runs/v2_push_3e5/checkpoints/000750/pretrained_model; DATA=data/lerobot_v3_d15_pp                  # v2_base -> 0919 push. eval 0.1560
# MODEL=runs/v2_push_3e5_aux/checkpoints/001500/pretrained_model; DATA=data/lerobot_v3_d15_pp              # 위 + aux. eval 0.1571
# MODEL=runs/v2_place_push_3e5to1e5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d16_all      # v2_base -> place -> push. eval 0.1437
# MODEL=runs/v2_place_push_3e5to1e5_aux/checkpoints/003250/pretrained_model; DATA=data/lerobot_v3_d16_all  # 위 + aux. eval 0.1457
# MODEL=runs/v2_push_place_3e5to1e5/checkpoints/001500/pretrained_model; DATA=data/lerobot_v3_d20_pu_pl    # v2_base -> push -> place. eval 0.1302
# MODEL=runs/v2_push_place_3e5to1e5_aux/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d20_pu_pl # 위 + aux. eval 0.1166

# --- [2026-10-06] loss 가중 lw0: base1(d10_new, relabel 없음), state noise p0.3 σ1.0 (train_lw0.sh) ---
# MODEL=runs/lw0_S_n/checkpoints/004750/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # 청크 스텝 0.9^k, noise σ1.0. eval 0.1088
# MODEL=runs/lw0_S_nn/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # 청크 스텝 0.9^k, noise 없음. eval 0.0909
# MODEL=runs/lw0_D_n/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # 관절 변화량, noise σ1.0. eval 0.1223
# MODEL=runs/lw0_D_nn/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # 관절 변화량, noise 없음. eval 0.1059
# MODEL=runs/lw0_E_n/checkpoints/004750/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # 에피소드 cos 1→0.5, noise σ1.0. eval 0.1126
# MODEL=runs/lw0_E_nn/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # 에피소드 cos 1→0.5, noise 없음. eval 0.0935
# MODEL=runs/lw0_SD_n/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # S+D, noise σ1.0. eval 0.1172
# MODEL=runs/lw0_SD_nn/checkpoints/005250/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # S+D, noise 없음. eval 0.1076
# MODEL=runs/lw0_SE_n/checkpoints/004750/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # S+E, noise σ1.0. eval 0.1086
# MODEL=runs/lw0_SE_nn/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # S+E, noise 없음. eval 0.0921
# MODEL=runs/lw0_DE_n/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # D+E, noise σ1.0. eval 0.1191
# MODEL=runs/lw0_DE_nn/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # D+E, noise 없음. eval 0.1077
# --- [2026-10-10] loss 가중 lw1: base1, state noise p0.3 σ0.1 (train_lw1.sh). 가중 없음+noise 없음 = d10_new ---
# MODEL=runs/lw1_C_n/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # 가중 없음, noise σ0.1. eval 0.0948
# MODEL=runs/lw1_S_n/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # 청크 스텝 0.9^k, noise σ0.1. eval 0.0919
# MODEL=runs/lw1_D_n/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # 관절 변화량, noise σ0.1. eval 0.1052
# MODEL=runs/lw1_W_n/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # 초반 24프레임 x4, noise σ0.1. eval 0.0945
# MODEL=runs/lw1_W_nn/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # 초반 24프레임 x4, noise 없음. eval 0.0955
# MODEL=runs/lw1_SD_n/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # S+D, noise σ0.1. eval 0.1033
# MODEL=runs/lw1_SW_n/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # S+W, noise σ0.1. eval 0.0954
# MODEL=runs/lw1_SW_nn/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # S+W, noise 없음. eval 0.0925
# MODEL=runs/lw1_DW_n/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # D+W, noise σ0.1. eval 0.1100
# MODEL=runs/lw1_DW_nn/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # D+W, noise 없음. eval 0.1109
# MODEL=runs/lw1_SDW_n/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # S+D+W, noise σ0.1. eval 0.1113
# MODEL=runs/lw1_SDW_nn/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # S+D+W, noise 없음. eval 0.1303

# --- [2026-10-11] pp1: lw1 레시피를 d15_pp(0912 place + 0919 push) 데이터로 (train_pp1.sh). 가중 없음+noise 없음 = d15_pp ---
# MODEL=runs/pp1_C_n/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d15_pp                      # 가중 없음, noise σ0.1. eval 0.0901
# MODEL=runs/pp1_W_n/checkpoints/005750/pretrained_model; DATA=data/lerobot_v3_d15_pp                      # 초반 24프레임 x4, noise σ0.1. eval 0.0962
# MODEL=runs/pp1_W_nn/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d15_pp                     # 초반 24프레임 x4, noise 없음. eval 0.0984
# MODEL=runs/pp1_SW_n/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d15_pp                     # S+W, noise σ0.1. eval 0.1015
# --- [2026-10-16] smolvla_ext 87판 체인 (run_ext_chain.sh). 전부 DATA=d9_0912, 홀드아웃 18개 고정 → d10_new(0.0934)와 비교된다 ---
#     이름 ext_<모듈>_<시작점>[_R<u|e>_a<알파>][_c<코다스텝>].  모듈 R=ROCKET(VGGT 교사 특징 정렬) / P=PVI / C=Coda(반복 보정)
#     시작점 W=smolvla_base부터(main AE lr 1e-4), F=d10_new 위 main AE 동결, L=d10_new 위 main AE 1e-5
#     Ru=VLM 3,7,11,15층을 교사 입력으로 / Re=실제로 일하는 0-3층 출력을.  α=ROCKET 손실 가중, c=Coda 스텝 수.
#     서버는 체크포인트의 smolvla_ext.json 을 보고 모듈을 붙인다(smolvla_server_openmani.py:589) -- 추가 플래그 없음.
#     아래 eval 은 verify.json 의 eval 최소 step. 틱 지표(gt 오차/산포/seam)는 results/ext_eval/<tag>@<step>.json.
#   R (12판, eval 오름차순)
# MODEL=runs/ext_R_F_Ru_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                       # R, d10_new, main AE 동결, Ru α0.125. eval 0.0914
# MODEL=runs/ext_R_L_Ru_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                         # R, d10_new, main AE 1e-5, Ru α0.5. eval 0.0924
# MODEL=runs/ext_R_L_Ru_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                       # R, d10_new, main AE 1e-5, Ru α0.125. eval 0.0929
# MODEL=runs/ext_R_F_Ru_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                         # R, d10_new, main AE 동결, Ru α0.5. eval 0.0929
# MODEL=runs/ext_R_F_Re_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                       # R, d10_new, main AE 동결, Re α0.125. eval 0.0930
# MODEL=runs/ext_R_W_Ru_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                       # R, base부터, main AE 1e-4, Ru α0.125. eval 0.0935
# MODEL=runs/ext_R_W_Ru_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                         # R, base부터, main AE 1e-4, Ru α0.5. eval 0.0937
# MODEL=runs/ext_R_W_Re_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                       # R, base부터, main AE 1e-4, Re α0.125. eval 0.0937
# MODEL=runs/ext_R_F_Re_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                         # R, d10_new, main AE 동결, Re α0.5. eval 0.0941
# MODEL=runs/ext_R_L_Re_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                       # R, d10_new, main AE 1e-5, Re α0.125. eval 0.0941
# MODEL=runs/ext_R_W_Re_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                         # R, base부터, main AE 1e-4, Re α0.5. eval 0.0949
# MODEL=runs/ext_R_L_Re_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                         # R, d10_new, main AE 1e-5, Re α0.5. eval 0.0951
#   P (3판, eval 오름차순)
# MODEL=runs/ext_P_W/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                                # P, base부터, main AE 1e-4. eval 0.0963
# MODEL=runs/ext_P_F/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                                # P, d10_new, main AE 동결. eval 0.0967
# MODEL=runs/ext_P_L/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                                # P, d10_new, main AE 1e-5. eval 0.0974
#   C (6판, eval 오름차순)
# MODEL=runs/ext_C_F_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                             # C, d10_new, main AE 동결, Coda 5. eval 0.0944
# MODEL=runs/ext_C_F_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                            # C, d10_new, main AE 동결, Coda 10. eval 0.0944
# MODEL=runs/ext_C_L_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                            # C, d10_new, main AE 1e-5, Coda 10. eval 0.0960
# MODEL=runs/ext_C_L_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                             # C, d10_new, main AE 1e-5, Coda 5. eval 0.0960
# MODEL=runs/ext_C_W_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                             # C, base부터, main AE 1e-4, Coda 5. eval 0.0981
# MODEL=runs/ext_C_W_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                            # C, base부터, main AE 1e-4, Coda 10. eval 0.1012
#   RP (12판, eval 오름차순)
# MODEL=runs/ext_RP_W_Ru_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                      # R+P, base부터, main AE 1e-4, Ru α0.125. eval 0.0940
# MODEL=runs/ext_RP_W_Re_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                      # R+P, base부터, main AE 1e-4, Re α0.125. eval 0.0940
# MODEL=runs/ext_RP_F_Ru_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                        # R+P, d10_new, main AE 동결, Ru α0.5. eval 0.0947
# MODEL=runs/ext_RP_L_Re_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                      # R+P, d10_new, main AE 1e-5, Re α0.125. eval 0.0951
# MODEL=runs/ext_RP_L_Ru_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                        # R+P, d10_new, main AE 1e-5, Ru α0.5. eval 0.0952
# MODEL=runs/ext_RP_W_Ru_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                        # R+P, base부터, main AE 1e-4, Ru α0.5. eval 0.0952
# MODEL=runs/ext_RP_L_Ru_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                      # R+P, d10_new, main AE 1e-5, Ru α0.125. eval 0.0953
# MODEL=runs/ext_RP_F_Ru_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                      # R+P, d10_new, main AE 동결, Ru α0.125. eval 0.0958
# MODEL=runs/ext_RP_F_Re_a0125/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                      # R+P, d10_new, main AE 동결, Re α0.125. eval 0.0958
# MODEL=runs/ext_RP_L_Re_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                        # R+P, d10_new, main AE 1e-5, Re α0.5. eval 0.0959
# MODEL=runs/ext_RP_F_Re_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                        # R+P, d10_new, main AE 동결, Re α0.5. eval 0.0962
# MODEL=runs/ext_RP_W_Re_a05/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                        # R+P, base부터, main AE 1e-4, Re α0.5. eval 0.0964
#   RC (24판, eval 오름차순)
# MODEL=runs/ext_RC_F_Ru_a0125_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+C, d10_new, main AE 동결, Ru α0.125, Coda 5. eval 0.0926
# MODEL=runs/ext_RC_L_Ru_a0125_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+C, d10_new, main AE 1e-5, Ru α0.125, Coda 5. eval 0.0930
# MODEL=runs/ext_RC_L_Ru_a05_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # R+C, d10_new, main AE 1e-5, Ru α0.5, Coda 5. eval 0.0936
# MODEL=runs/ext_RC_L_Ru_a0125_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+C, d10_new, main AE 1e-5, Ru α0.125, Coda 10. eval 0.0937
# MODEL=runs/ext_RC_F_Re_a0125_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+C, d10_new, main AE 동결, Re α0.125, Coda 5. eval 0.0941
# MODEL=runs/ext_RC_F_Re_a0125_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+C, d10_new, main AE 동결, Re α0.125, Coda 10. eval 0.0946
# MODEL=runs/ext_RC_F_Ru_a0125_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+C, d10_new, main AE 동결, Ru α0.125, Coda 10. eval 0.0947
# MODEL=runs/ext_RC_W_Re_a0125_c10/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+C, base부터, main AE 1e-4, Re α0.125, Coda 10. eval 0.0949
# MODEL=runs/ext_RC_L_Ru_a05_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+C, d10_new, main AE 1e-5, Ru α0.5, Coda 10. eval 0.0949
# MODEL=runs/ext_RC_L_Re_a0125_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+C, d10_new, main AE 1e-5, Re α0.125, Coda 5. eval 0.0950
# MODEL=runs/ext_RC_L_Re_a0125_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+C, d10_new, main AE 1e-5, Re α0.125, Coda 10. eval 0.0951
# MODEL=runs/ext_RC_F_Ru_a05_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # R+C, d10_new, main AE 동결, Ru α0.5, Coda 5. eval 0.0951
# MODEL=runs/ext_RC_F_Ru_a05_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+C, d10_new, main AE 동결, Ru α0.5, Coda 10. eval 0.0952
# MODEL=runs/ext_RC_W_Ru_a05_c10/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+C, base부터, main AE 1e-4, Ru α0.5, Coda 10. eval 0.0954
# MODEL=runs/ext_RC_W_Re_a05_c10/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+C, base부터, main AE 1e-4, Re α0.5, Coda 10. eval 0.0954
# MODEL=runs/ext_RC_W_Ru_a0125_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+C, base부터, main AE 1e-4, Ru α0.125, Coda 5. eval 0.0961
# MODEL=runs/ext_RC_L_Re_a05_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+C, d10_new, main AE 1e-5, Re α0.5, Coda 10. eval 0.0965
# MODEL=runs/ext_RC_L_Re_a05_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # R+C, d10_new, main AE 1e-5, Re α0.5, Coda 5. eval 0.0965
# MODEL=runs/ext_RC_F_Re_a05_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+C, d10_new, main AE 동결, Re α0.5, Coda 10. eval 0.0967
# MODEL=runs/ext_RC_W_Re_a0125_c5/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+C, base부터, main AE 1e-4, Re α0.125, Coda 5. eval 0.0969
# MODEL=runs/ext_RC_F_Re_a05_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # R+C, d10_new, main AE 동결, Re α0.5, Coda 5. eval 0.0969
# MODEL=runs/ext_RC_W_Ru_a0125_c10/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+C, base부터, main AE 1e-4, Ru α0.125, Coda 10. eval 0.0975
# MODEL=runs/ext_RC_W_Ru_a05_c5/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # R+C, base부터, main AE 1e-4, Ru α0.5, Coda 5. eval 0.0989
# MODEL=runs/ext_RC_W_Re_a05_c5/checkpoints/006000/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # R+C, base부터, main AE 1e-4, Re α0.5, Coda 5. eval 0.1000
#   PC (6판, eval 오름차순)
# MODEL=runs/ext_PC_F_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                           # P+C, d10_new, main AE 동결, Coda 10. eval 0.0938
# MODEL=runs/ext_PC_W_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                           # P+C, base부터, main AE 1e-4, Coda 10. eval 0.0940
# MODEL=runs/ext_PC_F_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                            # P+C, d10_new, main AE 동결, Coda 5. eval 0.0940
# MODEL=runs/ext_PC_L_c10/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                           # P+C, d10_new, main AE 1e-5, Coda 10. eval 0.0947
# MODEL=runs/ext_PC_L_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                            # P+C, d10_new, main AE 1e-5, Coda 5. eval 0.0948
# MODEL=runs/ext_PC_W_c5/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d9_0912                            # P+C, base부터, main AE 1e-4, Coda 5. eval 0.0948
#   RPC (24판, eval 오름차순)
# MODEL=runs/ext_RPC_W_Re_a05_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+P+C, base부터, main AE 1e-4, Re α0.5, Coda 10. eval 0.0941
# MODEL=runs/ext_RPC_L_Re_a0125_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # R+P+C, d10_new, main AE 1e-5, Re α0.125, Coda 10. eval 0.0948
# MODEL=runs/ext_RPC_L_Ru_a0125_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+P+C, d10_new, main AE 1e-5, Ru α0.125, Coda 5. eval 0.0948
# MODEL=runs/ext_RPC_W_Re_a0125_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # R+P+C, base부터, main AE 1e-4, Re α0.125, Coda 10. eval 0.0949
# MODEL=runs/ext_RPC_W_Re_a0125_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+P+C, base부터, main AE 1e-4, Re α0.125, Coda 5. eval 0.0955
# MODEL=runs/ext_RPC_F_Ru_a0125_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # R+P+C, d10_new, main AE 동결, Ru α0.125, Coda 10. eval 0.0959
# MODEL=runs/ext_RPC_L_Ru_a05_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+P+C, d10_new, main AE 1e-5, Ru α0.5, Coda 10. eval 0.0959
# MODEL=runs/ext_RPC_F_Re_a0125_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # R+P+C, d10_new, main AE 동결, Re α0.125, Coda 10. eval 0.0959
# MODEL=runs/ext_RPC_L_Ru_a0125_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # R+P+C, d10_new, main AE 1e-5, Ru α0.125, Coda 10. eval 0.0960
# MODEL=runs/ext_RPC_F_Ru_a0125_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+P+C, d10_new, main AE 동결, Ru α0.125, Coda 5. eval 0.0963
# MODEL=runs/ext_RPC_F_Re_a0125_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+P+C, d10_new, main AE 동결, Re α0.125, Coda 5. eval 0.0964
# MODEL=runs/ext_RPC_W_Ru_a0125_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                 # R+P+C, base부터, main AE 1e-4, Ru α0.125, Coda 10. eval 0.0965
# MODEL=runs/ext_RPC_L_Re_a0125_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+P+C, d10_new, main AE 1e-5, Re α0.125, Coda 5. eval 0.0966
# MODEL=runs/ext_RPC_W_Re_a05_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+P+C, base부터, main AE 1e-4, Re α0.5, Coda 5. eval 0.0966
# MODEL=runs/ext_RPC_F_Re_a05_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+P+C, d10_new, main AE 동결, Re α0.5, Coda 10. eval 0.0969
# MODEL=runs/ext_RPC_L_Ru_a05_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+P+C, d10_new, main AE 1e-5, Ru α0.5, Coda 5. eval 0.0969
# MODEL=runs/ext_RPC_F_Ru_a05_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+P+C, d10_new, main AE 동결, Ru α0.5, Coda 10. eval 0.0970
# MODEL=runs/ext_RPC_F_Ru_a05_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+P+C, d10_new, main AE 동결, Ru α0.5, Coda 5. eval 0.0971
# MODEL=runs/ext_RPC_L_Re_a05_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+P+C, d10_new, main AE 1e-5, Re α0.5, Coda 5. eval 0.0972
# MODEL=runs/ext_RPC_W_Ru_a0125_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                  # R+P+C, base부터, main AE 1e-4, Ru α0.125, Coda 5. eval 0.0973
# MODEL=runs/ext_RPC_W_Ru_a05_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+P+C, base부터, main AE 1e-4, Ru α0.5, Coda 5. eval 0.0977
# MODEL=runs/ext_RPC_F_Re_a05_c5/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                    # R+P+C, d10_new, main AE 동결, Re α0.5, Coda 5. eval 0.0978
# MODEL=runs/ext_RPC_L_Re_a05_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+P+C, d10_new, main AE 1e-5, Re α0.5, Coda 10. eval 0.0979
# MODEL=runs/ext_RPC_W_Ru_a05_c10/checkpoints/005000/pretrained_model; DATA=data/lerobot_v3_d9_0912                   # R+P+C, base부터, main AE 1e-4, Ru α0.5, Coda 10. eval 0.0993
#   eval 최소 상위 5: ext_R_F_Ru_a0125 0.0914, ext_R_L_Ru_a05 0.0924, ext_RC_F_Ru_a0125_c5 0.0926, ext_R_L_Ru_a0125 0.0929, ext_R_F_Ru_a05 0.0929


# --- [2026-09-27] 피라미드: base = d10_new(0912), 단계마다 4000스텝, 체인 안에서 lr 고정 (train_pyramid.sh) ---
# MODEL=runs/p_place_1e5/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919           # 0912 -> place, 1e-5. eval 0.0898 (= 기본값)
# MODEL=runs/p_place_3e5/checkpoints/000750/pretrained_model; DATA=data/lerobot_v3_d14_place0919           # 0912 -> place, 3e-5. eval 0.0917
# MODEL=runs/p_place_1e6/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919           # 0912 -> place, 1e-6. eval 0.1011
# MODEL=runs/p_push_1e5/checkpoints/003000/pretrained_model; DATA=data/lerobot_v3_d14_push0919             # 0912 -> push, 1e-5. eval 0.1565
# MODEL=runs/p_push_3e5/checkpoints/001000/pretrained_model; DATA=data/lerobot_v3_d14_push0919             # 0912 -> push, 3e-5. eval 0.1512
# MODEL=runs/p_push_1e6/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d14_push0919             # 0912 -> push, 1e-6. eval 0.2109
# MODEL=runs/p_push_place_1e5/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919      # 0912 -> push -> place, 1e-5. eval 0.0888
# MODEL=runs/p_push_place_3e5/checkpoints/000750/pretrained_model; DATA=data/lerobot_v3_d14_place0919      # 0912 -> push -> place, 3e-5. eval 0.0923
# MODEL=runs/p_push_place_1e6/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919      # 0912 -> push -> place, 1e-6. eval 0.1001
# MODEL=runs/p_place_push_1e5/checkpoints/003000/pretrained_model; DATA=data/lerobot_v3_d14_push0919       # 0912 -> place -> push, 1e-5. eval 0.1542
# MODEL=runs/p_place_push_3e5/checkpoints/001750/pretrained_model; DATA=data/lerobot_v3_d14_push0919       # 0912 -> place -> push, 3e-5. eval 0.1496
# MODEL=runs/p_place_push_1e6/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d14_push0919       # 0912 -> place -> push, 1e-6. eval 0.2070
# --- [2026-09-27] 피라미드, 2단에서 lr 을 낮춘 판 ---
# MODEL=runs/p_place_push_3e5to1e5/checkpoints/003000/pretrained_model; DATA=data/lerobot_v3_d14_push0919  # 0912 -> place(3e-5) -> push(1e-5). eval 0.1521
# MODEL=runs/p_place_push_1e5to1e6/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d14_push0919  # 0912 -> place(1e-5) -> push(1e-6). eval 0.2038
# MODEL=runs/p_push_place_3e5to1e5/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919 # 0912 -> push(3e-5) -> place(1e-5). eval 0.0922
# MODEL=runs/p_push_place_1e5to1e6/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919 # 0912 -> push(1e-5) -> place(1e-6). eval 0.1068

# --- [2026-09-22] 병합 후 한 번에 학습 (train_d19.sh) ---
# MODEL=runs/d19_full_drop/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d18_place_all         # 0912+0919 place 병합, 카메라 dropout 0.3. eval 0.0826
# MODEL=runs/d19_full/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d18_place_all              # 위와 같은 데이터, 증강 없음. eval 0.0839
# --- [2026-09-21] 0912 위에 0919 를 lr 1e-4 로 (train_d19.sh) ---
# MODEL=runs/d19_ft_place/checkpoints/003750/pretrained_model; DATA=data/lerobot_v3_d14_place0919          # 0912 -> 0919 place. eval 0.0976
# MODEL=runs/d19_ft_pp/checkpoints/004000/pretrained_model; DATA=data/lerobot_v3_d17_new_pp                # 0912 -> 0919 place+push 동시. eval 0.1083
# MODEL=runs/d19_ft_push/checkpoints/002000/pretrained_model; DATA=data/lerobot_v3_d14_push0919            # 0912 -> 0919 push. eval 0.1412
# --- [2026-09-20] 0919 를 처음 병합해 한 번에 학습 (train_d15.sh) ---
# MODEL=runs/d15_all/checkpoints/005250/pretrained_model; DATA=data/lerobot_v3_d16_all                     # 0912 place + 0919 place + push 566ep. eval 0.0840
# MODEL=runs/d15_pp/checkpoints/005500/pretrained_model; DATA=data/lerobot_v3_d15_pp                       # 0912 place + 0919 push 404ep. eval 0.0917
# --- [2026-09-13] 0912 만 (모든 p_* 의 base) ---
MODEL=runs/d10_new/checkpoints/004750/pretrained_model; DATA=data/lerobot_v3_d9_0912                     # 0912 place 242ep. eval 0.0934

# ================= 구 리그 (0905 이전). 카메라 위치가 달라 실기에 쓰지 말 것 =================
# MODEL=runs/d10_mix/checkpoints/006750/pretrained_model; DATA=data/lerobot_v3_d10_merged                  # d8(가중치 0.3) + d9 1970ep. eval 0.0792
# MODEL=runs/d8_base/checkpoints/006250/pretrained_model; DATA=data/lerobot_v3_d8_merged                   # d8 1728ep. eval 0.1692
# MODEL=runs/d8_mild/checkpoints/004500/pretrained_model; DATA=data/lerobot_v3_d8_merged                   # d8 + dropout 0.15 + smooth. eval 0.1753
# MODEL=runs/d8_multi/checkpoints/004250/pretrained_model; DATA=data/lerobot_v3_d7_0905                    # 0905 다물체만 648ep. eval 0.2538
# MODEL=runs/d5_G/checkpoints/003500/pretrained_model; DATA=data/lerobot_v3_d5_merged                      # d5 1080ep. eval 0.1785

# ================= 띄우지 말 것 =================
# runs/d12_pp (eval 0.0933) / runs/d12_all (eval 0.0872)
#   d15/d16 와 같은 데이터의 카메라 오염판이다. 0919 place 162개 전부와 push 81개에서
#   image_0/image_1 이 뒤바뀐 채 들어갔고 변환기는 image_0 을 camera1 로 위치 매핑하므로
#   d13_all 은 43%가 반대 시점이다. 게다가 뒤바뀜 여부가 place/push 와 거의 일치해서
#   모델이 문장 대신 그걸 지름길로 쓸 수 있다. 데이터셋(d12_pp/d13_all)은 이미 지웠다.
# runs/d6_* / d8_done
#   done_flag 6-D 계열. 2026-09-09 폐기, 종료 판정은 외부 VLM(Qwen)이 한다.
# runs/d3_prog
#   진행률 회귀. 실패로 확정.
# lerobot/smolvla_base  (웹 원본 가중치, HF hub. 로컬 캐시
#   ~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6...)
#   MODEL=lerobot/smolvla_base; DATA=data/lerobot_v3_d9_0912   <- 서버가 거부한다(2026-10-17 실측)
#   `refusing to start: 'observation.state' has NO normalization stats in the preprocessor
#    pipeline. Keys present: ['so100-blue.buffer.action','so100-red.buffer.action','so100.buffer.action']`
#   즉 원본 체크포인트에는 SO-100 의 통계만 있고 이 팔의 state/action 통계가 없다. 정규화가
#   조용히 통과되면 틱이 통째로 틀리므로 서버가 기동 자체를 막는다(게이트 2). 우회 플래그 없음.
#   그 밖에도 chunk_size=50, resize 512, action 6채널(마지막을 done_flag 로 오해한다)로
#   이 팔의 계약과 다르다. **파인튜닝의 출발점으로만 쓴다**:
#     train_smolvla_openmani.py --policy.path=lerobot/smolvla_base   (RUNBOOK.md:298)
#   웹 가중치에서 바로 올린 판을 띄우려면 위 목록의 ext_*_W(시작점 W, 29판) 또는 d10_new 를 쓴다.

# ================= 구 리그 (0905 이전). 입력 분포가 다르다 =================
# MODEL=runs/d8_base/checkpoints/006250/pretrained_model # d8 1728ep, 증강 없음. eval 0.1692
# DATA=data/lerobot_v3_d8_merged                         #   장면에 물체가 여러 개인 0905를 처음 포함 -> 명사 구분 테스트용
# MODEL=runs/d8_mild/checkpoints/004500/pretrained_model # 같은 d8 + 카메라 dropout 0.15 + smooth w=20. eval 0.1753
# DATA=data/lerobot_v3_d8_merged
# MODEL=runs/d8_multi/checkpoints/004250/pretrained_model # d7 648ep, 0905 다물체만. eval 0.2538
# DATA=data/lerobot_v3_d7_0905
# MODEL=runs/d5_G/checkpoints/003500/pretrained_model    # d5 1080ep, dropout + velocity. eval 0.1785
# DATA=data/lerobot_v3_d5_merged
# MODEL=runs/d3_*/checkpoints/...                        # d3 972ep. data/lerobot_v3_d3_joint 와 짝

# --- other checkpoints, by experiment ---------------------------------------
# eval_loss is listed for reference only; it does NOT rank these. A single run's eval
# oscillates +-0.004 every 250 steps -- the same size as the gaps between runs -- and both
# augmentations are switched off during eval anyway. The numbers that matter come from
# diag/variant_compare.py, quoted as the paired difference against B over 40 frames x 8 draws.
#
# 지표 뜻 (전부 틱):
#   draw 산포     같은 입력에 8번 물었을 때 답들이 서로 어긋나는 양 -> 떨림
#   gt 오차       예측 관절값 vs 시연자가 실제로 한 값 -> 정확도
#   seam          8스텝 실행 후 재예측했을 때 겹치는 구간의 불일치 -> 재계획 순간 튐
#   가림시 이동   카메라 한 대를 회색으로 덮었을 때 예측이 정상 대비 움직인 양
#
# A  chunk 8                runs/d3_A/checkpoints/006000/pretrained_model   eval 0.1025
# B  chunk 16, 기준선        runs/d3_B/checkpoints/006000/pretrained_model   eval 0.1505
# C  chunk 32                runs/d3_C/...   eval 0.2053   (DATA2로 아카이브)
# D  + lerobot 이미지 증강    runs/d3_D/...   eval 0.1423   (아카이브, 배포한 적 없음)
# E  + 카메라 dropout 0.15   runs/d3_E/checkpoints/006000/pretrained_model   eval 0.1470
#      가림시 이동 -3.91*, 가림시 오차 -1.43, 산포 +0.77*(약간 악화), 정확도 불변
# F  + 시간미분 w=6.5        runs/d3_F/checkpoints/006000/pretrained_model   eval 0.1525
#      산포 -2.64*, seam -1.88(무의미), 정확도 불변
# F2 + 시간미분 w=20         runs/d3_F2/checkpoints/006000/pretrained_model  eval 0.1610
#      산포 -4.59*, seam -7.27*, 가림시 오차 -3.20*, 그러나 가림시 이동 +1.74*(악화)
# G  = E + F2  (현행 권장)   runs/d3_G/checkpoints/006000/pretrained_model   eval 0.1558
#      산포 -4.39*, seam -4.61*, 가림시 이동 -3.33*, 가림시 오차 -4.56*, 정확도 -0.31
#      -> F2의 떨림 감소를 지키면서 F2가 망친 가림 강건성을 E 수준으로 되돌린다. 넷 중 최선.
#   (* = 2 표준오차 초과)
# prog 6채널 진행률 (실패)   runs/d3_prog/...  진행률 회귀는 실패로 확정. 띄우지 말 것.
#
# --- d5: 신규 108ep 병합 (2026-08-30) ---------------------------------------
# data/lerobot_v3_d5_merged = d3_joint 972ep + d4_0829 108ep = 1080ep / 125,979f / 54 task.
# 액션은 여전히 5-D(done 없음). d3의 972ep보다 에피소드가 1.85배 길고 1.15배 느리다.
# G  = E + F2                runs/d5_G/checkpoints/003500/pretrained_model   eval 0.1785
#      DATA=data/lerobot_v3_d5_merged 로 바꿔야 한다.
#
# --- d6: done_flag 채널 (2026-08-30) -- 폐기 ---------------------------------
# action[5] = done_flag 를 정책이 직접 회귀하던 계열. d6_A/E/V/G, d8_done 이 여기 속한다.
# 2026-09-09 폐기: 종료 판정은 외부 VLM(Qwen)이 맡는다. 이 체크포인트들은 띄우지 않는다.
#
# 주의: 프루너 버그로 A~F2의 eval 최소 체크포인트는 삭제됐다(step:5K 반올림 파싱).
#       지금 고쳐졌지만 지난 런은 복구 불가라 전부 step 6000으로 비교한다.

# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n lerobot python -u \
  smolvla_server_openmani.py \
    --model-path "$MODEL" \
    --dataset-root "$DATA" \
    --port 8000 \
    \
    `# --- 떨림 억제 --------------------------------------------------------` \
    `# 뽑기 N번 평균. 1->8에서 산포 611->261틱. 배치 한 번이라 지연은 거의 안 는다.` \
    --n-avg 1 \
    `# 청크 이음매 보정. 이음매 36.62 -> 10.65틱 (gt의 1스텝 이동 10.60과 거의 같음).` \
    --rtc \
    --rtc-execution-horizon 8 \
    --rtc-inference-delay 1 \
    `# --rtc-schedule LINEAR              ZEROS=머리만, ONES=구간 전체. 바꿔본 적 없음.` \
    `# --rtc-max-guidance-weight 10       가이던스 발산 상한. 기본값 유지 권장.` \
    `# --rtc-stale-reset-s 2              이 시간보다 오래된 이전 계획은 버린다.` \
    `# --no-rtc                           RTC 끄기.` \
    `# Coda 포함 체크포인트(ext_*C*)는 2단계로 샘플링하므로 RTC를 걸 자리가 갈린다. --coda-rtc 기본값` \
    `# auto 가 측정된 correction(보정 단계=실행되는 출력을 가이던스)을 고르고, 당김/스텝>1이면 가중치를` \
    `# 자동으로 낮춘다(c5 판 10->5). 따로 줄 필요 없음. 2026-10-17 측정(24쌍x2복제, EXPERIMENTS.md):` \
    `#   seam RTC off 20.93/21.42 -> main 13.55/13.11, correction 12.27/12.67, both 12.27/12.66 틱.` \
    `#   배치 간 차이는 쌍차 SE(2.1~3.4틱) 안이라 유의하지 않다. both는 35ms 더 비싸고 이득 없음.` \
    `# --coda-rtc main                    배치를 직접 고르기 (auto|off|main|correction|both)` \
    \
    `# --- 자유 문장 -> 정규 문장 매칭 (STT 앞단용) --------------------------` \
    `# 규칙 기반이 기본. 동사/물체/방향 슬롯을 키워드로 채우고, 못 채우면 어느 슬롯인지` \
    `# 말하고 거부한다. embed 모드(정책 자체 텍스트 모델 코사인)는 패러프레이즈 216개에서` \
    `# 58%%밖에 못 맞혀서(place/push 혼동) 비교용으로만 남겼다.` \
    `# --lang-snap` \
    `# --lang-snap-mode rules             또는 embed` \
    `# --lang-snap-min-sim 0.90           embed 모드에서만 의미 있음 [임의값, 미검증]` \
    `# --lang-snap-min-margin 0.005       embed 모드에서만 의미 있음 [임의값, 미검증]` \
    \
    `# --- 실행 구간 --------------------------------------------------------` \
    `# 응답에 실을 청크 스텝 수. 0 = 체크포인트의 8. 상한은 chunk_size(16).` \
    `# 청크 뒤 8스텝은 앞 8스텝의 1.70배 오차라 16은 손해다. 4로 줄이면 0.4초마다` \
    `# 재계획해서 더 정확하지만 왕복 시간이 그 안에 들어와야 한다(아래 slack 로그 참조).` \
    --emit-steps 8 \
    \
    `# --- 진단 / 안전 ------------------------------------------------------` \
    `# --- 종료 판정 --------------------------------------------------------` \
    `# 정책 내부 done 채널은 폐기(2026-09-09). 5-D 체크포인트라 서버는 항상 None을 내보내고,` \
    `# --done-threshold/--done-consecutive 는 6-D 전용이라 넘기지 않는다.` \
    `# 종료는 외부 VLM이 판정한다. 이 서버와 무관한 별도 프로세스다:` \
    `#   CUDA_VISIBLE_DEVICES=1 conda run -n lerobot python vlm_done_server.py --port 8100` \
    `# 클라이언트에 --vlm-done-url http://localhost:8100 을 주면 붙는다.` \
    `# 주의: 12에피소드에서 AUROC 0.57 / 0.56 / 0.38, 우연이 0.5다. 판별력 미확인.` \
    `# 처음에는 --no-stop-on-vlm-done 으로 점수만 보고 믿을 만한지부터 확인할 것.` \
    \
    --diag-out runs/deploy_probe.jsonl \
    `# --diag-every 1` \
    `# --attn-out runs/deploy_attn.jsonl   어텐션 덤프. 무겁다.` \
    `# --dry-run                           추론은 하되 응답에 execute=false. 배선 점검용.` \
    `# --allow-mismatch                    openmani_meta.json이 없거나 안 맞아도 진행. 평소 금지.` \
    `# --rate-limit-ticks 100              클리핑 예측에만 쓴다. 서버는 실제로 제한하지 않는다.` \
    `# --device cuda` \
    `# --host 0.0.0.0` \
    "$@"

# ============================================================================
# 클라이언트(로봇 PC)에서 대응되는 명령. RUNBOOK.md 1-2절에 전체가 있다.
#
#   python smolvla_client_local.py --server http://<이 머신 IP>:8000 \
#     --lang "push the knife to the right" --port /dev/ttyUSB0 \
#     --exec-steps 8 --step-dt-ms 100 --max-step-ticks 60 \
#     --smooth-actions --smooth-kernel 7 --smooth-sigma 1.0 \
#     --save-frames flagged --out-dir runs/rollout01
#
# 클라이언트가 직전 요청의 왕복 시간을 함께 보내므로, 서버 로그에
#   rtt=..ms slack=+..ms
# 가 찍힌다. slack = (실행할 스텝 수 x step-dt) - 왕복 시간.  음수면 팔이
# 계획을 다 쓰고 놀았다는 뜻이라 --emit-steps를 더 줄이면 안 된다.
# 종료 시 서버가 왕복 시간 중앙/p90/최대와 slack 요약을 출력한다.
# ============================================================================
