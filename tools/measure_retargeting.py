"""Measure accepted retargeting runs and write the presentation table."""
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "topo"))
from compare import load_demo, make_retargeter
from variants import DemoRetargeter, PENETRATION_CAP

REFERENCE = ROOT / "data/retargeted_traj/wujihand/screwdriver/retargeted_120fps.h5"
OUTPUT = ROOT / "demo_assets/02_retarget_comparison.md"
LOGS = ROOT / "logs"
RUNS = (
    ("ik", "Keypoint IK"),
    ("baseline", "REGRIND + Clarabel"),
    ("weighted", "Weighted Laplacian"),
    ("soft", "Soft penetration only"),
)
NEAR = 0.01


class Measurements:
    def __init__(self):
        self.retargeter = make_retargeter(DemoRetargeter, variant="topo")
        hand = self.retargeter
        self.finger_by_geometry = {}
        for index in hand.plant.GetBodyIndices(hand.wrist.body().model_instance()):
            body = hand.plant.get_body(index)
            if body.name().startswith("right_finger"):
                finger = body.name().split("_")[1]
                for geometry in hand.plant.GetCollisionGeometriesForBody(body):
                    self.finger_by_geometry[geometry] = finger

    def collision_depths(self, q):
        """Return peak hand-object, inter-finger, and all hand self-overlap in meters."""
        hand = self.retargeter
        hand.plant.SetPositions(hand.plant_context, q)
        object_overlap = finger_overlap = self_overlap = 0.0
        for pair in hand.query_object.ComputeSignedDistancePairwiseClosestPoints(0.0):
            geometries = {pair.id_A, pair.id_B}
            if geometries & hand.table_geometries:
                continue
            depth = -pair.distance
            if geometries & hand.object_geometries:
                object_overlap = max(object_overlap, depth)
            else:
                self_overlap = max(self_overlap, depth)
                first = self.finger_by_geometry.get(pair.id_A)
                second = self.finger_by_geometry.get(pair.id_B)
                if first and second and first != second:
                    finger_overlap = max(finger_overlap, depth)
        return object_overlap, finger_overlap, self_overlap

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


def report(topo_run=None):
    _, human, objects, _ = load_demo()
    measurement = Measurements()
    source = human[:, measurement.retargeter.mano_mapped_joint_indices]
    motions = []
    times = []
    fallbacks = []
    runs = RUNS if topo_run is None else (*RUNS, (topo_run, "TopoRetarget-lite (soft self)"))
    for variant, _ in runs:
        with np.load(LOGS / f"{variant}.npz") as data:
            q = data["motion"]
            if q.shape != (len(human), measurement.retargeter.nq) or not np.isfinite(q).all():
                raise ValueError(f"{variant} did not produce a complete finite trajectory")
            np.testing.assert_allclose(q[:, -7:], objects, atol=1e-10)
            motions.append(q)
            times.append(float(data["elapsed"]) * 1000 / len(human))
            rate = int(data["rejected_seeds"]) / len(human)
            if rate >= 0.05:
                raise ValueError(f"{variant} exceeds the fallback threshold ({rate:.1%})")
            fallbacks.append(rate)

    near = np.array([measurement.near_contact(q, points) for q, points in zip(motions[1], source)])
    if not near.any():
        raise ValueError("No source points meet the near-contact threshold")

    keypoints = [measurement.keypoints(motion) for motion in motions]
    depths = np.array([[measurement.collision_depths(q) for q in motion] for motion in motions])
    if topo_run is not None and depths[-1].max() > PENETRATION_CAP + 1e-6:
        raise ValueError(f"{topo_run} exceeded the penetration cap in the scored trajectory")
    penetration = 1000 * depths[:, :, 0]
    peak_finger_overlap = 1000 * depths[:, :, 1].max(axis=1)
    object_rot = Rotation.from_quat(objects[:, :4], scalar_first=True).as_matrix()
    drifts = [1000 * np.linalg.norm(np.einsum("tji,tkj->tki", object_rot, points - source), axis=2)
              for points in keypoints]
    mean_drift = [drift[near].mean() for drift in drifts]
    np.savez(LOGS / "retarget_metrics.npz", penetration=penetration,
             drift=np.array(drifts), near=near, run_ids=[name for name, _ in runs],
             labels=[label for _, label in runs])

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

    rows = []
    for i, (_, label) in enumerate(runs):
        depth = penetration[i]
        overlap = "" if topo_run is None else f" {peak_finger_overlap[i]:.2f} |"
        rows.append(
            f"| {label} | {fallbacks[i]:.1%} | {depth.mean():.2f} | {depth.max():.2f} | "
            f"{100 * (depth > 2).mean():.1f}% |{overlap} {mean_drift[i]:.2f} | {times[i]:.1f} |"
        )
    overlap_header = "" if topo_run is None else "Max finger self-penetration (mm) | "
    overlap_rule = "" if topo_run is None else "---:|"
    overlap_definition = (
        "Finger self-penetration is the largest overlap Drake reports between distinct fingers across the sequence. "
        if topo_run is not None else ""
    )
    interpretation = (
        f"Mean near-contact drift was {mean_drift[1]:.2f} mm with released REGRIND, "
        f"{mean_drift[2]:.2f} mm with weighted Laplacian, and {mean_drift[3]:.2f} mm with soft penetration alone. "
        f"Soft penetration alone completed every frame and reached {penetration[3].max():.2f} mm peak overlap. "
        "It retains REGRIND’s uniform interaction graph and objective weights and uses the original slack penalty. "
        "The replay uses the recorded screwdriver motion in every panel."
    )
    full_variant = (
        f"TopoRetarget-lite with soft self-collision had {fallbacks[-1]:.1%} fallbacks, "
        f"{mean_drift[-1]:.2f} mm mean drift, and {peak_finger_overlap[-1]:.2f} mm peak "
        f"finger-to-finger overlap. The original hard-self-collision run fell back on 834 of 1,081 frames."
        if topo_run else
        "The original full TopoRetarget-lite run fell back on 834 of 1,081 frames (77.2%). "
        "The full-variant retries also exceeded the 5% threshold. See PROGRESS.md for the feasibility probes."
    )
    OUTPUT.write_text(
        "# WujiHand screwdriver retargeting comparison\n\n"
        f"One REGRIND demonstration, {len(human):,} frames at 120 fps. All runs used Drake Clarabel on the Mac.\n\n"
        "| Variant | Fallback rate | Mean penetration (mm) | Max penetration (mm) | Frames over 2 mm | "
        f"{overlap_header}Mean near-contact drift (mm) | Mean solve time (ms/frame) |\n"
        f"|---|---:|---:|---:|---:|{overlap_rule}---:|---:|\n" + "\n".join(rows) + "\n\n"
        + interpretation + " " + full_variant + "\n\n"
        "Fallback counts a frame if its initial refinement fails and restarts from the previous collision-aware pose. "
        "The comparison includes runs below 5%. "
        "Penetration is the deepest signed-distance overlap of a hand–screwdriver collision pair "
        "in Drake’s collision geometry, with zero for a frame without overlap. "
        f"{overlap_definition}"
        "The drift is the mean Euclidean error in screwdriver "
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topo-run")
    args = parser.parse_args()
    report(args.topo_run)
