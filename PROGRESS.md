# Presentation demo progress

- [ ] Part 2. Colab L4 provisioning blocked. No training or Colab replay.
- [x] Part 3. Local comparison and MuJoCo renders complete.
- [ ] Part 4. Waiting for “morning”.

## Part 2 — Colab

- This checkout was on `main` with upstream `origin`. Created `presentation-demo` and pushed to `ReyZix/regrind` via remote `fork`.
- Installed Colab CLI 0.8.0 under Python 3.12. The project `.envrc` was absent. Both local ADC attempts returned “Backend rejected accelerator 'L4'”, and CLI OAuth required browser consent. `colab --auth adc stop -s trainer` found no session. No training ETA. REGRIND is configured for 20,000 iterations.

## Part 3 — local demo

- Python 3.11 at `~/work/.venv`, Drake 1.40 with Clarabel, and MuJoCo 4.0.1. The released baseline ran directly from unchanged REGRIND source and reproduced the shipped trajectory to 0.000133 mm mean keypoint error.
- Soft penetration alone retained 100 object samples, uniform Laplacian weights, and REGRIND's objective weights. It completed all 1,081 frames with zero fallbacks at `1e5`. Finite differences at four contact frames agreed with Drake's distance derivative. A fixed-pose check required 0.014 m of slack for −0.015 m distance and rejected −0.031 m distance under the 0.029 m slack cap.
- The full TopoRetarget-lite run fell back on 834/1,081 frames. The bone-only solve succeeded at frames 58–60, and refinement returned `kInfeasibleConstraints`. Removing just the hard hand self-collision constraints made those same subproblems feasible with the object and table constraints intact. The diagnostic did not alter the production constraints.
- The first 150 frames already exceeded the 54-fallback allowance for the full sequence in all three attempts. Lower penalties `1e3` and `1e4` each had 92 fallbacks. The squared hinge with a separate hard distance bound had 92, including four solver-specific errors. Using REGRIND's interaction and smoothness weights with its zero bone refinement weight had 111. The full variant was excluded from the comparison.
- Reran IK, baseline, weighted Laplacian, and soft-only across all 1,081 frames with zero fallbacks. Weighted Laplacian had the lowest mean near-contact drift (4.78 mm). Soft-only had 1.01 mm peak penetration and 5.34 mm drift. Regenerated the table, verified the 271-frame comparison video decodes end to end, and inspected the keyframe at 1.33 seconds.
- Reproduce the four accepted runs with `~/work/.venv/bin/python topo/compare.py VARIANT` for `baseline`, `ik`, `weighted`, and `soft`. Then run `tools/measure_retargeting.py` and `tools/render_comparison.py comparison` with the same interpreter. Use `tools/render_comparison.py reference` for the shipped backup.
- Reproduce the full-variant pilots with `topo/compare.py topo --frames 150 --probe-failures` and one of `--slack-penalty 1000`, `--slack-penalty 10000`, `--hinge`, or `--baseline-weights`. Use a unique `--run-id` for each. Per-frame solver statuses and raw trajectories live in ignored `logs/`.
