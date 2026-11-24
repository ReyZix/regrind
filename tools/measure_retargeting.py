"""Measure the four retargeted trajectories and write the presentation table."""
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "topo"))
from compare import VARIANTS, load_demo, make_retargeter
from variants import DemoRetargeter

REFERENCE = ROOT / "data/retargeted_traj/wujihand/screwdriver/retargeted_120fps.h5"
OUTPUT = ROOT / "demo_assets/02_retarget_comparison.md"
LOGS = ROOT / "logs"
LABELS = ("Keypoint IK", "Released REGRIND + Clarabel", "REGRIND + weighted Laplacian", "TopoRetarget-lite")
NEAR = 0.01


class Measurements:
    def __init__(self):
        self.retargeter = make_retargeter(DemoRetargeter, variant="topo")

    def penetration(self, q):
        hand = self.retargeter
        hand.plant.SetPositions(hand.plant_context, q)
        pairs = hand.query_object.ComputeSignedDistancePairwiseClosestPoints(0.0)
        return max(
            (-pair.distance for pair in pairs
             if bool({pair.id_A, pair.id_B} & hand.object_geometries)
             and not {pair.id_A, pair.id_B} & hand.table_geometries),
            default=0.0,
        )

    def near_contact(self, q, source):
        hand = self.retargeter
        hand.plant.SetPositions(hand.plant_context, q)
        near = []
        for point in source:
            distances = hand.query_object.ComputeSignedDistanceToPoint(point, NEAR)
            signed_distance = min(
                (result.distance for result in distances if result.id_G in hand.object_geometries),
                default=np.inf,
            )
            near.append(abs(signed_distance) <= NEAR)
        return near

    def keypoints(self, mot):
        hand = self.retargeter
        names = hand.laplacian_match_links.values()
        return np.array([hand._get_robot_link_positions(q, names) for q in mot])


def report():
    _, human, objects, _ = load_demo()
    measurement = Measurements()
    source = human[:, measurement.retargeter.mano_mapped_joint_indices]
    motions = []
    times = []
    rejected_seeds = 0
    for variant in VARIANTS:
        with np.load(LOGS / f"{variant}.npz") as data:
            q = data["motion"]
            if q.shape != (len(human), measurement.retargeter.nq) or not np.isfinite(q).all():
                raise ValueError(f"{variant} did not produce a complete finite trajectory")
            np.testing.assert_allclose(q[:, -7:], objects, atol=1e-10)
            motions.append(q)
            times.append(float(data["elapsed"]) * 1000 / len(human))
            if variant == "topo":
                rejected_seeds = int(data["rejected_seeds"])

    near = np.array([measurement.near_contact(q, points) for q, points in zip(motions[1], source)])
    if not near.any():
        raise ValueError("No source points meet the near-contact threshold")

    keypoints = [measurement.keypoints(motion) for motion in motions]
    penetration = np.array([[1000 * measurement.penetration(q) for q in motion] for motion in motions])
    object_rot = Rotation.from_quat(objects[:, :4], scalar_first=True).as_matrix()
    drifts = [1000 * np.linalg.norm(np.einsum("tji,tkj->tki", object_rot, points - source), axis=2)
              for points in keypoints]
    mean_drift = [drift[near].mean() for drift in drifts]
    np.savez(LOGS / "retarget_metrics.npz", penetration=penetration,
             drift=np.array(drifts), near=near)

    baseline = motions[1]
    with h5py.File(REFERENCE) as ref:
        reference_names = ref["mano_keypoint_names"].asstr()[:].tolist()
        reference_order = [reference_names.index(name) for name in measurement.retargeter.laplacian_match_links]
        reference_points = ref["robot_keypoints"][:][:, reference_order]
        wrist_mm = 1000 * np.linalg.norm(baseline[:, 4:7] - ref["robot_pos"][:], axis=1)
        wrist_degrees = np.rad2deg((Rotation.from_quat(baseline[:, :4], scalar_first=True).inv() *
                                    Rotation.from_quat(ref["robot_quat"][:], scalar_first=True)).magnitude())
        joint_degrees = np.rad2deg(baseline[:, 7:27] - ref["robot_joints"][:])
        kp_mm = 1000 * np.linalg.norm(keypoints[1] - reference_points, axis=2)
        object_mm = 1000 * np.linalg.norm(objects[:, 4:] - ref["object_pos"][:], axis=1)
        object_degrees = np.rad2deg((Rotation.from_quat(objects[:, :4], scalar_first=True).inv() *
                                     Rotation.from_quat(ref["object_quat"][:], scalar_first=True)).magnitude())
        source_mm = 1000 * np.linalg.norm(human - ref["mano_joint_coords"][:], axis=2)

    rows = [
        f"| {label} | {depth.max():.2f} | {100 * (depth > 2).mean():.1f}% | {drift:.2f} | {elapsed:.1f} |"
        for label, depth, drift, elapsed in zip(LABELS, penetration, mean_drift, times)
    ]
    weighted_change = mean_drift[2] - mean_drift[1]
    topo_change = mean_drift[3] - mean_drift[1]
    weighted_direction = "lower" if weighted_change < 0 else "higher"
    topo_direction = "lower" if topo_change < 0 else "higher"
    interpretation = (
        f"Relative to released REGRIND, the weighted Laplacian produced {abs(weighted_change):.2f} mm "
        f"{weighted_direction} mean contact drift and TopoRetarget-lite produced {abs(topo_change):.2f} mm "
        f"{topo_direction} drift. The slack variant allowed up to {penetration[3].max():.2f} mm "
        f"measured overlap, compared with {penetration[1].max():.2f} mm for the hard-constraint baseline. "
        "TopoRetarget-lite also changes the sample count and adds bone matching and stronger temporal "
        "regularization, so its row measures the combined effect on this demonstration. "
        f"Its bone-only seed caused an infeasible subproblem on {rejected_seeds:,} of {len(human):,} frames. "
        "Those frames started refinement again from the previous collision-aware pose, with the bone prior active. "
        "The replay uses the recorded screwdriver motion in every panel."
    )
    OUTPUT.write_text(
        "# WujiHand screwdriver retargeting comparison\n\n"
        f"One REGRIND demonstration, {len(human):,} frames at 120 fps. All four runs used Drake Clarabel on the Mac.\n\n"
        "| Variant | Max penetration (mm) | Frames over 2 mm | Mean near-contact drift (mm) | Mean solve time (ms/frame) |\n"
        "|---|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        + interpretation + "\n\n"
        "Penetration is the deepest signed-distance overlap of a hand–screwdriver collision pair "
        "in Drake’s collision geometry. The drift is the mean Euclidean error in screwdriver "
        "coordinates over mapped source keypoints within 10 mm of the collision surface "
        f"({near.sum():,} point-frames). Solve time is elapsed retargeting time per frame, "
        "including geometry and optimization, measured before scoring and rendering. The local "
        "variants assemble a smaller quadratic program, so time comparisons also reflect implementation differences.\n\n"
        "## Shipped trajectory check\n\n"
        f"The released retargeting code with Clarabel differs from the shipped trajectory by "
        f"{wrist_mm.mean():.6f} mm mean wrist translation, {wrist_degrees.mean():.6f}° mean "
        f"wrist rotation, {np.sqrt(np.mean(joint_degrees**2)):.6f}° joint-angle RMSE, and "
        f"{kp_mm.mean():.6f} mm mean mapped keypoint error. The source keypoints differ by "
        f"{source_mm.mean():.3f} mm on average. The screwdriver poses differ by "
        f"{object_mm.mean():.3f} mm and {object_degrees.mean():.3f}° on average.\n\n"
        "Method and selected parameters are from [TopoRetarget, Sections 3.2–3.4 and Appendix A.1]"
        "(https://arxiv.org/html/2606.16272v2).\n"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    report()
