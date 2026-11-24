# Project: REGRIND on Colab + TopoRetarget-lite on Mac (presentation tomorrow)

## Machines
- Local: MacBook M4, macOS, no NVIDIA GPU. Isaac Sim cannot run locally.
- Cloud: Google Colab via the colab CLI (https://github.com/googlecolab/google-colab-cli).
  Session name "trainer", GPU L4 only (Isaac Sim needs RT cores; never A100/H100).

## Goals (priority order)
1. Colab: train Regrind-WujiHand-Screwdriver-v0 exactly as released.
2. Mac: TopoRetarget-lite retargeting comparison (guaranteed demo).
3. Outputs go to demo_assets/, committed on branch presentation-demo, pushed from the Mac.

## Rules
- Economical usage: tail/grep logs, no polling loops, short reports.
- Hard stop after 3 distinct failed fix attempts per step; diagnose in PROGRESS.md.
- On Colab failure or completion: colab stop -s trainer.
- Clarabel solver only, never MOSEK. Don't edit REGRIND's original files; new code in topo/, tools/.
- Commit small files only. Never commit .envrc, credentials, datasets, checkpoint histories.

## TopoRetarget changes to implement (as options, default = REGRIND behavior)
1. Laplacian weights w_ij = exp(-30 * ||v_i - v_j||), row-normalized per vertex, computed on the
   human source configuration (hand keypoints + object samples) and reused for the robot.
2. Soft penetration: signed distance >= -tau - s, 0 <= s <= b - tau, tau = 0.001 m, b = 0.03 m,
   slack penalty 1e5 * s^2 / 2.
3. Bone-direction initialization (match differences of adjacent finger-bone directions plus a
   smoothness term), only if time allows.
Other TopoRetarget parameters: 50 object samples, lambda_IM = 500, lambda_bone = 0.1, lambda_reg = 2.5.
