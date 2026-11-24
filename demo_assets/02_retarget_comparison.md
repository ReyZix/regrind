# WujiHand screwdriver retargeting comparison

One REGRIND demonstration, 1,081 frames at 120 fps. All runs used Drake Clarabel on the Mac.

| Variant | Fallback rate | Mean penetration (mm) | Max penetration (mm) | Frames over 2 mm | Mean near-contact drift (mm) | Mean solve time (ms/frame) |
|---|---:|---:|---:|---:|---:|---:|
| Keypoint IK | 0.0% | 2.03 | 7.59 | 47.5% | 4.88 | 0.8 |
| REGRIND + Clarabel | 0.0% | 0.68 | 1.00 | 0.0% | 5.34 | 140.8 |
| Weighted Laplacian | 0.0% | 0.69 | 1.00 | 0.0% | 4.78 | 38.8 |
| Soft penetration only | 0.0% | 0.69 | 1.01 | 0.0% | 5.34 | 45.4 |

Mean near-contact drift was 5.34 mm with released REGRIND, 4.78 mm with weighted Laplacian, and 5.34 mm with soft penetration alone. Soft penetration alone completed every frame and reached 1.01 mm peak overlap. It retains REGRIND’s uniform interaction graph and objective weights and uses the original slack penalty. The replay uses the recorded screwdriver motion in every panel. The original full TopoRetarget-lite run fell back on 834 of 1,081 frames (77.2%). The full-variant retries also exceeded the 5% threshold. See PROGRESS.md for the feasibility probes.

Fallback counts a frame if its initial refinement fails and restarts from the previous collision-aware pose. The comparison includes runs below 5%. Penetration is the deepest signed-distance overlap of a hand–screwdriver collision pair in Drake’s collision geometry, with zero for a frame without overlap. The drift is the mean Euclidean error in screwdriver coordinates over mapped source keypoints within 10 mm of the collision surface (2,550 point-frames). Solve time is elapsed retargeting time per frame, including geometry and optimization, measured before scoring and rendering. The local variants assemble a smaller quadratic program, so time comparisons also reflect implementation differences.

## Shipped trajectory check

The released retargeting code with Clarabel differs from the shipped trajectory by 0.000104 mm mean wrist translation, 0.000160° mean wrist rotation, 0.000270° joint-angle RMSE, and 0.000133 mm mean mapped keypoint error. The source keypoints differ by 0.000 mm on average. The screwdriver poses differ by 0.000 mm and 0.000° on average.

Method and selected parameters are from [TopoRetarget, Sections 3.2–3.4 and Appendix A.1](https://arxiv.org/html/2606.16272v2).
