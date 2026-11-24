"""Run the WujiHand screwdriver comparison with Drake Clarabel."""
import argparse
import os
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from pydrake.solvers import ClarabelSolver

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/regrind")]
os.environ.setdefault("REGRIND_DATA_DIR", str(ROOT / "data"))

from scripts.retarget_hand_object import interpolate_demo, load_custom_zed_mocap_demo
from regrind.retargeting import wujihand_constants as wuji
from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter

CONFIG = wuji.get_object_config("screwdriver")
VARIANTS = ("ik", "baseline", "weighted", "soft", "topo")


def make_retargeter(cls=HandInteractionMeshOneStageRetargeter, **options):
    return cls(
        robot_model_path=wuji.ROBOT_URDF_FILE,
        robot_name=wuji.ROBOT_NAME,
        object_model_path=CONFIG["object_urdf_file"],
        object_name="screwdriver",
        table_height=CONFIG["table_height"],
        demo_joints=wuji.MANO_JOINTS,
        laplacian_match_links=wuji.MANO_TO_WUJI_MAPPING,
        q_a_init_idx=-7,
        step_size=0.2,
        visualize=False,
        debug=False,
        w_nominal_tracking_init=5.0,
        solver=ClarabelSolver(),
        **options,
    )


def load_demo():
    arrays = load_custom_zed_mocap_demo(CONFIG["demo_file"])
    return interpolate_demo(*arrays, factor=4)


def abort_failed_baseline_solve():
    raise RuntimeError("The released retargeter reported a failed Clarabel solve.")


def run(variant, frame_limit=None, *, run_id=None, **solver_options):
    wrist, human, objects, _ = load_demo()
    human, objects = human[:frame_limit], objects[:frame_limit]
    initial = np.r_[wrist[0], np.zeros(wuji.ROBOT_DOF)]
    sample_file = CONFIG["object_keypoints_paths"]["bottom"]
    if variant == "topo":
        sample_file = ROOT / "data/keypoints/screwdriver_50.npy"
    samples = np.load(sample_file)
    suffix = "-smoke" if frame_limit is not None else ""
    output = ROOT / "logs" / f"{run_id or variant}{suffix}.npz"
    output.parent.mkdir(exist_ok=True)
    rejected_seeds = 0
    if variant == "baseline":
        retargeter = make_retargeter()
        start = perf_counter()
        motion, *_ = retargeter.retarget_motion(
            human_joint_motions=human,
            object_poses=objects,
            object_poses_augmented=objects.copy(),
            object_points_local_demo=samples,
            object_points_local=samples,
            q_a_init=initial,
        )
    else:
        from variants import DemoRetargeter

        retargeter = make_retargeter(DemoRetargeter, variant=variant, **solver_options)
        start = perf_counter()
        with output.with_suffix(".jsonl").open("w") as frame_log:
            motion, rejected_seeds = retargeter.solve_demo(human, objects, samples, initial, wrist, frame_log)
    elapsed = perf_counter() - start

    np.savez(output, motion=motion, elapsed=elapsed, rejected_seeds=rejected_seeds)
    print(f"{variant}: {len(motion)} frames in {elapsed:.2f} s → {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=VARIANTS, default="baseline")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--slack-penalty", type=float, default=1e5)
    parser.add_argument("--self-slack-penalty", type=float,
                        help="Enable bounded hand self-collision slack in TopoRetarget-lite")
    parser.add_argument("--hinge", action="store_true")
    parser.add_argument("--baseline-weights", action="store_true")
    parser.add_argument("--probe-failures", action="store_true")
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    if args.run_id is not None and not re.fullmatch(r"[a-z0-9_-]+", args.run_id):
        parser.error("--run-id must use lowercase letters, digits, hyphens or underscores")
    if not np.isfinite(args.slack_penalty) or args.slack_penalty <= 0:
        parser.error("--slack-penalty must be positive and finite")
    if args.self_slack_penalty is not None:
        if args.variant != "topo":
            parser.error("--self-slack-penalty requires the topo variant")
        if not np.isfinite(args.self_slack_penalty) or args.self_slack_penalty <= 0:
            parser.error("--self-slack-penalty must be positive and finite")
    sys.breakpointhook = abort_failed_baseline_solve
    run(args.variant, args.frames, run_id=args.run_id, slack_penalty=args.slack_penalty,
        self_slack_penalty=args.self_slack_penalty,
        hinge=args.hinge, baseline_weights=args.baseline_weights, probe_failures=args.probe_failures)
