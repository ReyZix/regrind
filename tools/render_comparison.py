"""Render fixed-state WujiHand screwdriver trajectories with MuJoCo."""
import argparse
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "topo"))
from compare import CONFIG, VARIANTS, make_retargeter, wuji

ASSETS = ROOT / "demo_assets"
LOGS = ROOT / "logs"
REFERENCE = ROOT / "data/retargeted_traj/wujihand/screwdriver/retargeted_120fps.h5"
LABELS = ("Keypoint IK", "REGRIND", "Weighted Laplacian", "TopoRetarget-lite")
ACCENTS = ("#a7bbd2", "#78c7f1", "#93d8bd", "#f6c477")
FONT_PATH = Path("/System/Library/Fonts/Avenir Next.ttc")


def font(size, bold=False):
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size, index=0 if bold else 7)
    return ImageFont.load_default(size=size)


def reference_motion():
    with h5py.File(REFERENCE) as data:
        return np.column_stack([
            data[key][:] for key in ("robot_quat", "robot_pos", "robot_joints", "object_quat", "object_pos")
        ])


def build_model(width, height):
    mjcf = Path(wuji.ROBOT_URDF_FILE).with_suffix(".xml")
    xml = ET.parse(mjcf).getroot()
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "headlight", ambient="0.38 0.38 0.41", diffuse="0.78 0.78 0.78")
    ET.SubElement(visual, "rgba", haze="0.085 0.12 0.17 1")
    assets = xml.find("asset")
    for mesh in assets.findall("mesh"):
        mesh.set("file", str((mjcf.parent / mesh.get("file")).resolve()))
    ET.SubElement(
        assets, "mesh", name="screwdriver_visual",
        file=str(Path(CONFIG["object_urdf_file"]).with_name("screwdriver.obj")),
    )
    ET.SubElement(assets, "texture", name="background", type="skybox", builtin="gradient",
                  rgb1="0.13 0.19 0.27", rgb2="0.055 0.085 0.13", width="256", height="256")
    world = xml.find("worldbody")
    world.find("body").set("mocap", "true")
    obj = ET.SubElement(world, "body", name="screwdriver", mocap="true")
    ET.SubElement(obj, "geom", type="mesh", mesh="screwdriver_visual",
                  rgba="1 0.47 0.09 1", contype="0", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


class Scene:
    def __init__(self, width, height):
        self.model = build_model(width, height)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 130
        self.camera.elevation = -23
        self.camera.distance = 0.41
        plant = make_retargeter().plant
        self.joints = []
        for name in [f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]:
            self.joints.append((self.model.joint(name).qposadr[0], plant.GetJointByName(name).position_start()))
        self.hand_id = self.model.body("root").mocapid[0]
        self.object_id = self.model.body("screwdriver").mocapid[0]

    def render(self, q, focus):
        for address, index in self.joints:
            self.data.qpos[address] = q[index]
        self.data.mocap_quat[self.hand_id], self.data.mocap_pos[self.hand_id] = q[:4], q[4:7]
        self.data.mocap_quat[self.object_id], self.data.mocap_pos[self.object_id] = q[-7:-3], q[-3:]
        self.camera.lookat[:] = focus
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        return Image.fromarray(self.renderer.render())

    def close(self):
        self.renderer.close()


def camera_path(motion):
    rotation = Rotation.from_quat(motion[:, -7:-3], scalar_first=True)
    tool_middle = rotation.apply([0, 0.064, 0]) + motion[:, -3:]
    focus = 0.72 * tool_middle + 0.28 * motion[:, 4:7]
    return gaussian_filter1d(focus, sigma=9, axis=0, mode="nearest")


def compose(scene, motions, focuses, i, comparison):
    if comparison:
        canvas = Image.new("RGB", (1280, 752), "#101923")
        layout = ((24, 73), (648, 73), (24, 405), (648, 405))
        title = "WujiHand screwdriver retargeting"
        labels, accents = LABELS, ACCENTS
    else:
        canvas = Image.new("RGB", (960, 608), "#101923")
        layout = ((24, 73),)
        title = "WujiHand screwdriver reference"
        labels, accents = ("Shipped REGRIND trajectory",), (ACCENTS[1],)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 14), title, fill="#f1f6fb", font=font(31, True))
    draw.text((canvas.width - 24, 29), f"{i / 120:04.1f} s", fill="#b9cbdc",
              font=font(18), anchor="ra")
    for (x, y), motion, label, accent in zip(layout, motions, labels, accents):
        panel = scene.render(motion[i], focuses[i])
        canvas.paste(panel, (x, y + 31))
        draw.rectangle((x, y, x + panel.width - 1, y + 30), fill="#1c2b3c")
        draw.rectangle((x, y, x + 4, y + 30), fill=accent)
        draw.text((x + 14, y + 4), label, fill="#eff5fb", font=font(18, True))
    return canvas


def render(comparison, preview=False):
    if comparison:
        motions = [np.load(LOGS / f"{variant}.npz")["motion"] for variant in VARIANTS]
        if len({len(motion) for motion in motions}) != 1:
            raise ValueError("Comparison trajectories must have identical lengths")
        with np.load(LOGS / "retarget_metrics.npz") as metrics:
            near = metrics["near"]
            drift = metrics["drift"]
        gap = np.where(near, abs(drift[1] - drift[3]), 0.0).max(axis=1)
        keyframe = int(np.argmax(gaussian_filter1d(gap, sigma=6)))
        filename = "03_side_by_side.mp4"
        scene = Scene(608, 290)
        focus = camera_path(motions[1])
    else:
        motions = [reference_motion()]
        filename = "05_reference_mujoco.mp4"
        scene = Scene(912, 480)
        focus = camera_path(motions[0])
    try:
        if preview:
            i = min(280, len(motions[0]) - 1)
            compose(scene, motions, focus, i, comparison).save(LOGS / "render-preview.png")
            return
        if comparison:
            compose(scene, motions, focus, keyframe, comparison).save(ASSETS / "04_keyframe.png")
            print(f"Keyframe selected at {keyframe / 120:.2f} s (frame {keyframe})", flush=True)
        with imageio.get_writer(ASSETS / filename, fps=30, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for i in range(0, len(motions[0]), 4):
                writer.append_data(np.asarray(compose(scene, motions, focus, i, comparison)))
    finally:
        scene.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("comparison", "reference"))
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    render(args.mode == "comparison", args.preview)
