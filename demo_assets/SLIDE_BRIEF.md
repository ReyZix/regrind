# REGRIND and TopoRetarget-lite slide brief

## 1. REPO FACTS

- Fork [ReyZix/regrind](https://github.com/ReyZix/regrind), branch `presentation-demo`.
- Latest commit at the results snapshot, before this brief, `3d6cef251626a37a8fb2b1dd0cfe7c51916fc20f`.

| Milestone | Commit |
|---|---|
| Upstream REGRIND release | `38347a9e30184620df04e19c63c7c72378cae103` |
| Presentation scaffolding | `daf66c894ef6881031acaca390ea0ffcf9a29cfb` |
| Initial local comparison and renders | `110765535eae5bc7d0c91c83ac46365d4b3e950b` |
| Fallback diagnosis and soft-only control | `b63e1304c828ccaff6f162e951172f62491c8061` |
| Soft self-collision validated and five-panel demo added | `415f74753b60dab496e707ed9d0e2a8823a2b533` |
| Isolated Colab authentication and training runner | `66b72c899458f431ff8f6d10283d8ca4ff0156e9` |
| Colab authentication and runtime checks | `21be8317b2ecadd5a4fd6e4d43ebe13bc95ed1f0` |
| High-memory T4 allocation failure recorded | `3d6cef251626a37a8fb2b1dd0cfe7c51916fc20f` |

- Local run hardware and environment were Apple M4, Python **3.11.16**, MuJoCo **4.0.1**, and Drake **1.40.0**. All optimization runs used Drake's **Clarabel** solver.
- The same host reports macOS **27.2.1**, build **26B131**, at briefing time. The exact macOS version at run time is **UNKNOWN** because it was absent from the run logs.

## 2. BASELINE VERIFICATION

- Robot and object were the right WujiHand and screwdriver. Source was `data/zed_mocap_demo/screwdriver/demo_30fps.h5`, with 271 frames at 30 fps and 21 MANO keypoints. Fourfold interpolation yielded **1,081 frames at 120 fps**.
- The released retargeter ran with Clarabel. For each saved baseline pose in `logs/baseline.npz`, Drake forward kinematics produced world-frame robot keypoints, which were aligned by MANO name with `robot_keypoints` in `data/retargeted_traj/wujihand/screwdriver/retargeted_120fps.h5`.
- Euclidean error over **22,701** paired keypoints was **0.000132663994908 mm mean** and **0.014842440109456 mm max**, computed from the saved baseline poses and shipped keypoints. The comparison report rounds the mean to **0.000133 mm**.
- The checked-in report also lists **0.000104 mm** mean wrist translation error, **0.000160°** mean wrist rotation error, and **0.000270°** joint-angle RMSE.

## 3. VARIANTS

The released baseline is the reference for all changes below. The local variants use a reduced quadratic program and Drake's kinematic Jacobian. They retain joint limits, use a step-radius setting of **0.2** after frame 0, and allow up to **50** refinement steps at frame 0 and **10** on later frames. The full variants also use those iteration limits for bone warm starts.

| Variant | Change and parameters | Implementation |
|---|---|---|
| Keypoint IK | World-frame Euclidean keypoint residual with weight **10** and temporal regularization **0.5**. Object terms, interaction graph, and collision constraints are disabled. | `topo/variants.py` (`DemoRetargeter.refine`, `solve_demo`) |
| REGRIND + Clarabel | Released uniform Laplacian with **100** object samples, interaction weight **10**, smoothness **0.5**, and hard collision tolerance **τ = 0.001 m**. Clarabel is passed explicitly. | `topo/compare.py` (`make_retargeter`, `run`), released `retargeter.py` (`retarget_motion`, `solve_single_iteration`) |
| Weighted Laplacian | Same **100** samples, weights **10** and **0.5**, and hard **τ = 0.001 m**. Each Delaunay neighbor receives **exp(−30d)** with **κ = 30 m⁻¹**, normalized per vertex on the source and reused for the robot. | `topo/variants.py` (`interaction_laplacian`, `refine`, `solve_demo`) |
| Soft penetration only | Uniform Laplacian, **100** samples, weights **10** and **0.5**. Adds hand–object slack with **τ = 0.001 m**, **b = 0.030 m**, **0 ≤ s ≤ 0.029 m**, and cost **(10⁵/2)s²**. Hand self-collision and hand–table contact stay hard. | `topo/variants.py` (`interaction_laplacian`, `add_collision_constraints`, `refine`) |
| Original TopoRetarget-lite (excluded) | Weighted Laplacian with **κ = 30 m⁻¹**, **50** object samples, interaction weight **500/N** over **N = 71** vertices, and adjacent unit-bone direction differences. Bone warm-start weight **1.0**, refinement bone weight **0.1**, temporal regularization **2.5**. Hand–object slack uses **τ = 0.001 m**, **b = 0.030 m**, and penalty **10⁵**. Hand self-collision and hand–table contact stay hard. | `topo/compare.py` (`run`), `topo/variants.py` (`interaction_laplacian`, `bone_differences`, `add_collision_constraints`, `refine`, `solve_demo`) |
| TopoRetarget-lite (soft self) | Original full configuration plus hand self-collision slack, with its own **τ = 0.001 m**, **b = 0.030 m**, **0 ≤ s ≤ 0.029 m**, and **(10⁵/2)s²** penalty. Hand–table contact stays hard. | `topo/compare.py` (`--self-slack-penalty`), `topo/variants.py` (`add_collision_constraints`, `refine`, `solve_demo`) |

The four excluded 150-frame pilots kept the full configuration except for their tested change. Two used hand–object penalties **10³** and **10⁴**. One used an equivalent squared hinge with coefficient **10⁵/2** and a separate **φ ≥ −0.030 m** bound. One used REGRIND interaction weight **10**, smoothness **0.5**, and zero bone refinement weight while keeping the bone warm start. Pilot switches are handled in `topo/compare.py` and `topo/variants.py` (`add_collision_constraints`, `refine`).

## 4. RESULTS TABLE

| Variant | Fallback rate | Mean penetration (mm) | Max penetration (mm) | Frames over 2 mm | Max finger self-penetration (mm) | Mean near-contact drift (mm) | Mean solve time (ms/frame) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keypoint IK | 0.0% | 2.03 | 7.59 | 47.5% | 5.18 | 4.88 | 0.8 |
| REGRIND + Clarabel | 0.0% | 0.68 | 1.00 | 0.0% | 1.00 | 5.34 | 140.8 |
| Weighted Laplacian | 0.0% | 0.69 | 1.00 | 0.0% | 1.00 | 4.78 | 38.8 |
| Soft penetration only | 0.0% | 0.69 | 1.01 | 0.0% | 1.00 | 5.34 | 45.4 |
| TopoRetarget-lite (soft self) | 2.7% | 0.86 | 1.10 | 0.0% | 1.17 | 19.71 | 286.1 |

**Metric definitions**

- **Fallback rate** is the percentage of frames whose initial refinement failed and was retried from the previous collision-aware pose.
- **Mean penetration (mm)** averages each frame's deepest hand–screwdriver overlap from Drake's signed collision distances, counting nonoverlapping frames as zero.
- **Max penetration (mm)** takes the largest of those per-frame depths across the trajectory.
- **Frames over 2 mm** is the percentage of frames whose deepest hand–screwdriver overlap strictly exceeds 2 mm.
- **Max finger self-penetration (mm)** is the deepest signed-distance overlap between collision geometries belonging to two distinct fingers across all queried frames.
- **Mean near-contact drift (mm)** is the mean Euclidean distance between corresponding robot and human keypoints in screwdriver coordinates over the **2,550** mapped source keypoint-frames whose absolute signed distance to the screwdriver collision surface is at most **10 mm**.
- **Mean solve time (ms/frame)** is elapsed retargeting wall time divided by frame count, including geometry and optimization before scoring or rendering.

**Percentage change vs. REGRIND + Clarabel**

Computed from the displayed values as 100 × (variant − REGRIND) / REGRIND, rounded to two decimal places. **UNKNOWN** denotes division by the baseline's zero rate. For context, IK exceeded 2 mm on **47.5%** of frames, and the accepted full variant retried **29/1,081** frames.

| Variant | Fallback rate | Mean penetration | Max penetration | Frames over 2 mm | Max finger self-penetration | Mean near-contact drift | Mean solve time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keypoint IK | UNKNOWN | +198.53% | +659.00% | UNKNOWN | +418.00% | −8.61% | −99.43% |
| REGRIND + Clarabel | UNKNOWN | 0.00% | 0.00% | UNKNOWN | 0.00% | 0.00% | 0.00% |
| Weighted Laplacian | UNKNOWN | +1.47% | 0.00% | UNKNOWN | 0.00% | −10.49% | −72.44% |
| Soft penetration only | UNKNOWN | +1.47% | +1.00% | UNKNOWN | 0.00% | 0.00% | −67.76% |
| TopoRetarget-lite (soft self) | UNKNOWN | +26.47% | +10.00% | UNKNOWN | +17.00% | +269.10% | +103.20% |
| Original TopoRetarget-lite (excluded archive) | UNKNOWN | +29.41% | +12.00% | UNKNOWN | 0.00% | +232.21% | −43.96% |

The excluded row was summarized from the archived 1,081 poses in `logs/topo.npz` at the same display precision. The original full run had **77.2%** fallbacks, **0.88 mm** mean and **1.12 mm** max hand–object penetration, **0.0%** over 2 mm, **1.00 mm** max finger self-penetration, **17.74 mm** drift, and **78.9 ms/frame**. The 150-frame pilots use a different window.

## 5. FAILURES AND DIAGNOSES

- **Hard hand self-collision.** The original full variant first hit an infeasible subproblem at frame **58** and retried **834/1,081** frames. The bone-only solve succeeded at frames **58–60**. Removing only hard hand self-collision constraints made all three refinement subproblems feasible with the object and table constraints intact.
- **Pilot fixes.** Penalties **10³** and **10⁴** each produced **92/150** fallbacks. The squared hinge produced **92/150**, including four solver-specific errors. REGRIND objective weights produced **111/150**. Each exceeded the **54** fallbacks allowed across the full sequence. The soft-penetration-only control passed **1,081/1,081** frames without fallback at **10⁵**.
- **Soft self-collision fix.** The full run with self penalty **10⁵** had **29/1,081** retries, all after Clarabel solver-specific errors, and every retry solved. Its peak hand–object overlap was **1.10 mm** and its peak finger–finger overlap was **1.17 mm**. It passed the **<5%** fallback gate.
- **L4.** The original ambient credentials drew accelerator rejection. After scoped login, L4 allocation exhausted HTTP read-timeout retries and a separate **240-second** read-timeout attempt. The allocation cause is **UNKNOWN**.
- **High-memory T4.** The allocation returned **HTTP 503** initially and with backoff retries. The cause is **UNKNOWN**.
- **Standard T4.** SSH confirmed a Tesla T4 with **15,360 MiB** GPU memory, **12 GiB** system RAM, and **zero swap**. Released before setup due to the documented **32 GB** system RAM minimum. [NVIDIA Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- **Colab outcome.** Last verified **472.04 compute units**, **zero** active assignments, no training or cloud replay, and training throughput and ETA **UNKNOWN**. Released training configuration specifies **20,000** iterations.

## 6. WHAT I BUILT VS. WHAT I REUSED

**Added or changed on the fork** relative to the upstream release. Counts are text lines at this snapshot.

| File | Change | Lines |
|---|---|---:|
| `.gitignore` | Modified | 95 |
| `CLAUDE.md` | Added | 28 |
| `PROGRESS.md` | Added | 32 |
| `bin/colab` | Added, project ADC wrapper | 11 |
| `topo/compare.py` | Added, retargeting runner | 114 |
| `topo/variants.py` | Added, retargeting variants | 252 |
| `tools/measure_retargeting.py` | Added, metric extraction | 195 |
| `tools/render_comparison.py` | Added, MuJoCo comparison and replay | 174 |
| `tools/run_training.py` | Added, Colab runner and checkpoint sync | 184 |
| `demo_assets/02_retarget_comparison.md` | Added | 21 |
| `demo_assets/SLIDE_BRIEF.md` | Added | 143 |
| Other media listed below | Added | Binary |

**REGRIND files used unchanged**

- `scripts/retarget_hand_object.py` for demo loading and interpolation, plus `source/regrind/regrind/data/utils.py` for HDF5 loading.
- `source/regrind/regrind/retargeting/{retargeter,drake_utils,wujihand_constants}.py` for the baseline optimizer, plant and collision setup, Delaunay graph, and MANO-to-WujiHand mapping.
- `data/zed_mocap_demo/screwdriver/demo_30fps.h5`, `data/keypoints/screwdriver_{50,100}.npy`, and `data/retargeted_traj/wujihand/screwdriver/retargeted_120fps.h5`.
- Under `source/regrind/regrind/assets/`, the WujiHand `mjcf/free_wuji_right_hand_full_simplified_v1.{urdf,xml}` and referenced meshes, plus the `screwdriver/screwdriver.urdf`, collision meshes, and `screwdriver/screwdriver.obj` visual mesh.

## 7. HONEST CAVEATS

- The experiment covers one trajectory, one right WujiHand model, and one screwdriver task.
- Near-contact drift and penetration are local metrics. TopoRetarget reports ContactPose-specific metrics on other data. [TopoRetarget paper](https://arxiv.org/html/2606.16272v2)
- The full local variant changes the sample count, objective weights, warm start, and collision handling together. It met the retry gate while increasing mean drift from **5.34 mm** to **19.71 mm**.
- The videos play back poses with the same recorded object motion in every panel. They do not measure grasp stability, policy performance, or hardware transfer.
- Solve times include differences in how the released and local variants assemble the quadratic program.

## 8. MEDIA INVENTORY

| Path under `demo_assets/` | Bytes | Contents |
|---|---:|---|
| `02_retarget_comparison.md` | 2,642 | Final five-variant comparison and baseline verification |
| `03_side_by_side.mp4` | 4,564,286 | 9.033008 s, 1280 × 752, 271 frames at 30 fps. Top row, left to right — Keypoint IK, REGRIND + Clarabel, Weighted Laplacian. Bottom row, left to right — Soft penetration only, TopoRetarget-lite (soft self). |
| `04_keyframe.png` | 151,312 | 1280 × 752, five-panel keyframe at source frame 658 |
| `05_reference_mujoco.mp4` | 1,817,779 | 9.033333 s, 960 × 608, 271 frames at 30 fps. One panel — Shipped REGRIND trajectory. |
| `SLIDE_BRIEF.md` | 14,781 | This brief |
| `slides/slide_keyframe.png` | 151,312 | 1280 × 752, exact copy of the selected five-panel keyframe |
| `slides/slide_stills_1.png` | 173,017 | 1280 × 752, comparison at 0.000000 s |
| `slides/slide_stills_2.png` | 216,443 | 1280 × 752, comparison at 5.500000 s, nearest to the chosen contact keyframe |
| `slides/slide_stills_3.png` | 210,640 | 1280 × 752, comparison at 9.000000 s |
| `slides/slide_results_chart.png` | 95,915 | 2000 × 937, white-background bar charts of mean drift and max hand–object penetration in mm |
| `slides/slide_reference.png` | 192,505 | 1280 × 811, shipped REGRIND replay at 5.500000 s, scaled from 960 × 608 |

## 9. IMAGES FOR SLIDES

- Use `slide_keyframe.png` for the largest smoothed spread in near-contact error. The render log gives **5.48 s** at source frame **658** (zero-based).
- The three comparison stills are FFmpeg exports of video frames **0**, **165**, and **270** (zero-based). Each retains the panel labels.
- `slide_results_chart.png` uses the published comparison values. `slide_reference.png` is an FFmpeg export of reference video frame **165**, upscaled with Lanczos.
