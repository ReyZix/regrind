"""Clarabel variants for the released WujiHand screwdriver example."""

import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import MathematicalProgram
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from regrind.retargeting.drake_utils import create_interaction_mesh, get_adjacency_list
from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter

FINGERS = ((1, 2, 3, 17), (4, 5, 6, 18), (7, 8, 9, 20), (10, 11, 12, 19), (13, 14, 15, 16))
TAU = 0.001
PENETRATION_CAP = 0.03


def interaction_laplacian(source):
    _, tetrahedra = create_interaction_mesh(source)
    matrix = np.zeros((len(source), len(source)))
    for i, neighbors in enumerate(get_adjacency_list(tetrahedra, len(source))):
        if neighbors:
            distances = np.linalg.norm(source[neighbors] - source[i], axis=1)
            weights = np.exp(-30 * distances)
            matrix[i, i] = 1
            matrix[i, neighbors] = -weights / weights.sum()
    return matrix


def bone_differences(points, jacobians):
    edges = np.diff(points, axis=1)
    lengths = np.linalg.norm(edges, axis=2, keepdims=True)
    directions = edges / lengths
    differences = np.diff(directions, axis=1).ravel()
    if jacobians is None:
        return differences, None
    projectors = np.eye(3) - directions[..., :, None] * directions[..., None, :]
    derivatives = (projectors / lengths[..., None]) @ np.diff(jacobians, axis=1)
    return differences, np.diff(derivatives, axis=1).reshape(-1, jacobians.shape[-1])


def add_residual(prog, variables, jacobian, residual, weight):
    hessian = 2 * weight * (jacobian.T @ jacobian)
    gradient = 2 * weight * (jacobian.T @ residual)
    constant = weight * (residual @ residual)
    prog.AddQuadraticCost(hessian, gradient, constant, variables, is_convex=True)


class DemoRetargeter(HandInteractionMeshOneStageRetargeter):
    def __init__(self, *, variant, **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.frames = [
            self.plant.GetBodyByName(link).body_frame()
            for link in self.laplacian_match_links.values()
        ]
        self.fingers = np.array([
            [self.mano_mapped_joint_indices.index(k) for k in finger] for finger in FINGERS
        ])
        self.wrist = self.plant.GetBodyByName("base").body_frame()
        self.object = self.plant.GetBodyByName("bottom").body_frame()
        self.object_geometries = set(
            self.plant.GetCollisionGeometriesForBody(self.plant.GetBodyByName("bottom"))
        )
        self.table_geometries = set(
            self.plant.GetCollisionGeometriesForBody(self.plant.GetBodyByName("table_link"))
        )

    def linearize_keypoints(self, reference):
        positions = []
        jacobians = []
        for frame in self.frames:
            positions.append(frame.CalcPose(self.plant_context, reference).translation())
            jacobian = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context, JacobianWrtVariable.kQDot,
                frame, np.zeros(3), reference, reference,
            )
            jacobians.append(jacobian[:, self.q_a_indices])
        return np.array(positions), np.array(jacobians)

    def add_collision_constraints(self, prog, delta, q):
        jacobians, distances = self._update_jacobians_and_phis_from_q(q)
        for pair, distance in distances.items():
            object_pair = self.object_geometries.intersection(pair)
            if object_pair and self.table_geometries.intersection(pair):
                continue
            jacobian = jacobians[pair][self.q_a_indices]
            variables = delta
            if self.variant == "topo" and object_pair:
                slack = prog.NewContinuousVariables(1, "penetration")
                prog.AddBoundingBoxConstraint(0.0, PENETRATION_CAP - TAU, slack)
                prog.AddQuadraticCost(np.array([[1e5]]), np.zeros(1), slack)
                jacobian = np.r_[jacobian, 1.0]
                variables = np.r_[delta, slack]
            prog.AddLinearConstraint(jacobian, -distance - TAU, np.inf, variables)

    def refine(self, q, previous, target, samples, laplacian, source_bones, *, first, warm=False):
        last_cost = np.inf
        for _ in range(50 if first else 10):
            self.plant.SetPositions(self.plant_context, q)
            prog = MathematicalProgram()
            delta = prog.NewContinuousVariables(self.nq_a, "delta")
            lower, upper = self.q_a_lb - q[:self.nq_a], self.q_a_ub - q[:self.nq_a]
            if warm:
                lower[:7] = upper[:7] = 0
            prog.AddBoundingBoxConstraint(lower, upper, delta)
            if not first:
                prog.AddLorentzConeConstraint(np.r_[self.step_size, delta])

            if not warm:
                reference = self.plant.world_frame() if self.variant == "ik" else self.object
                points, jacobians = self.linearize_keypoints(reference)
                if laplacian is None:
                    residual = (points - target).ravel()
                    jacobian = jacobians.reshape(-1, self.nq_a)
                    weight = 10
                else:
                    displacement = np.vstack([points, samples]) - target
                    hand_operator = laplacian[:, :len(points)]
                    residual = (laplacian @ displacement).ravel()
                    jacobian = (hand_operator @ jacobians.reshape(len(points), -1)).reshape(-1, self.nq_a)
                    weight = 500 / len(laplacian) if self.variant == "topo" else self.laplacian_weights
                add_residual(prog, delta, jacobian, residual, weight)
                if self.variant != "ik":
                    self.add_collision_constraints(prog, delta, q)
            if source_bones is not None:
                local, derivatives = self.linearize_keypoints(self.wrist)
                directions, jacobian = bone_differences(local[self.fingers], derivatives[self.fingers])
                add_residual(prog, delta, jacobian, directions - source_bones, 1.0 if warm else 0.1)

            smoothness = 2.5 if self.variant == "topo" else self.smooth_weight
            prog.AddQuadraticErrorCost(
                smoothness * np.eye(self.nq_a), previous[:self.nq_a] - q[:self.nq_a], delta
            )
            result = self.solver.Solve(prog)
            if not result.is_success():
                raise RuntimeError(f"Clarabel failed in {self.variant} ({result.get_solution_result()})")
            q[:self.nq_a] += result.GetSolution(delta)
            q[:4] /= np.linalg.norm(q[:4])
            cost = result.get_optimal_cost()
            if np.isclose(cost, last_cost, atol=1e-10):
                break
            last_cost = cost
        return q

    def solve_demo(self, human, objects, samples, initial, wrists):
        previous = np.r_[initial, objects[0]]
        previous_warm = previous.copy()
        solutions = []
        rejected_seeds = 0
        for i in tqdm(range(len(human)), desc=self.variant):
            source = human[i, self.mano_mapped_joint_indices]
            pose = objects[i]
            q = np.r_[previous[:self.nq_a], pose]
            laplacian = None
            source_bones = None
            if self.variant != "ik":
                object_rotation = Rotation.from_quat(pose[:4], scalar_first=True)
                source = np.vstack([object_rotation.inv().apply(source - pose[4:]), samples])
                laplacian = interaction_laplacian(source)
            if self.variant == "topo":
                wrist_rotation = Rotation.from_quat(wrists[i, :4], scalar_first=True)
                local = wrist_rotation.inv().apply(human[i] - wrists[i, 4:])
                mapped = local[self.mano_mapped_joint_indices]
                source_bones, _ = bone_differences(mapped[self.fingers], None)
                warm = np.r_[wrists[i], previous_warm[7:self.nq_a], pose]
                q = self.refine(warm, previous_warm, None, samples, None, source_bones, first=i == 0, warm=True)
                previous_warm = q.copy()
                # Continue refining from the previous optimized wrist pose.
                q[:7] = previous[:7]
            try:
                q = self.refine(q, previous, source, samples, laplacian, source_bones, first=i == 0)
            except RuntimeError as error:
                if self.variant != "topo" or i == 0:
                    raise RuntimeError(f"Frame {i}: {error}") from error
                # A bone-only seed can make the next collision linearization infeasible.
                # Retry from the last collision-aware pose with this frame's object pose.
                rejected_seeds += 1
                q = np.r_[previous[:self.nq_a], pose]
                try:
                    q = self.refine(q, previous, source, samples, laplacian, source_bones, first=False)
                except RuntimeError as retry_error:
                    raise RuntimeError(f"Frame {i}, after rejecting bone seed: {retry_error}") from retry_error
            solutions.append(q)
            previous = q
        if rejected_seeds:
            print(f"Used the previous collision-aware pose on {rejected_seeds} frames.", flush=True)
        return np.array(solutions), rejected_seeds
