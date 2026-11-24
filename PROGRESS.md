# Presentation demo progress

- [ ] Part 2. Colab L4 provisioning blocked. No training or Colab replay.
- [x] Part 3. Local comparison and MuJoCo renders complete.
- [ ] Part 4. Waiting for “morning”.

## Part 2 — Colab

- This checkout was on `main` with upstream `origin`. Created `presentation-demo` and pushed to `ReyZix/regrind` via remote `fork`.
- Installed Colab CLI 0.8.0 under Python 3.12. The project `.envrc` was absent. Both local ADC attempts returned “Backend rejected accelerator 'L4'”, and CLI OAuth required browser consent. `colab --auth adc stop -s trainer` found no session. No training ETA. REGRIND is configured for 20,000 iterations.

## Part 3 — local demo

- Python 3.11 at `~/work/.venv`, Drake 1.40 with Clarabel, and MuJoCo 4.0.1. The released baseline ran directly from unchanged REGRIND source and reproduced the shipped trajectory to 0.000133 mm mean keypoint error.
- All four runs completed the 1,081 frames. The first TopoRetarget-lite run hit an infeasible collision linearization at frame 58. The successful retry restarted 834 frames from the previous collision-aware pose, with the bone objective still active.
- Mean near-contact drift in mm was 4.88 (IK), 5.34 (REGRIND), 4.78 (weighted), and 17.74 (TopoRetarget-lite). The three object-aware runs had zero frames over 2 mm penetration. See `demo_assets/02_retarget_comparison.md` for all metrics.
- Verified the comparison video and shipped-reference backup decode end to end. Both run 9.03 seconds at 30 fps. The keyframe shows the grip at 2.58 seconds.
- Reproduce with `~/work/.venv/bin/python topo/compare.py VARIANT` for `baseline`, `ik`, `weighted`, and `topo`. Then run `tools/measure_retargeting.py` and `tools/render_comparison.py comparison` with the same interpreter. Use `tools/render_comparison.py reference` for the shipped backup. Raw trajectories and logs live in ignored `logs/`.
