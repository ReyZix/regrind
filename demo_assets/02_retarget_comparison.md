# WujiHand screwdriver retargeting comparison

One REGRIND demonstration, 1,081 frames at 120 fps. All four runs used Drake Clarabel on the Mac.

| Variant | Max penetration (mm) | Frames over 2 mm | Mean near-contact drift (mm) | Mean solve time (ms/frame) |
|---|---:|---:|---:|---:|
| Keypoint IK | 7.59 | 47.5% | 4.88 | 0.8 |
| Released REGRIND + Clarabel | 1.00 | 0.0% | 5.34 | 133.5 |
| REGRIND + weighted Laplacian | 1.00 | 0.0% | 4.78 | 36.2 |
| TopoRetarget-lite | 1.12 | 0.0% | 17.74 | 78.9 |

Relative to released REGRIND, the weighted Laplacian produced 0.56 mm lower mean contact drift and TopoRetarget-lite produced 12.40 mm higher drift. The slack variant allowed up to 1.12 mm measured overlap, compared with 1.00 mm for the hard-constraint baseline. TopoRetarget-lite also changes the sample count and adds bone matching and stronger temporal regularization, so its row measures the combined effect on this demonstration. Its bone-only seed caused an infeasible subproblem on 834 of 1,081 frames. Those frames started refinement again from the previous collision-aware pose, with the bone prior active. The replay uses the recorded screwdriver motion in every panel.

Penetration is the deepest signed-distance overlap of a hand–screwdriver collision pair in Drake’s collision geometry. The drift is the mean Euclidean error in screwdriver coordinates over mapped source keypoints within 10 mm of the collision surface (2,550 point-frames). Solve time is elapsed retargeting time per frame, including geometry and optimization, measured before scoring and rendering. The local variants assemble a smaller quadratic program, so time comparisons also reflect implementation differences.

## Shipped trajectory check

The released retargeting code with Clarabel differs from the shipped trajectory by 0.000104 mm mean wrist translation, 0.000160° mean wrist rotation, 0.000270° joint-angle RMSE, and 0.000133 mm mean mapped keypoint error. The source keypoints differ by 0.000 mm on average. The screwdriver poses differ by 0.000 mm and 0.000° on average.

Method and selected parameters are from [TopoRetarget, Sections 3.2–3.4 and Appendix A.1](https://arxiv.org/html/2606.16272v2).
