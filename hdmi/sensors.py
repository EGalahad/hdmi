"""Scene-owned sensors for HDMI tasks.

``hdmi_head_camera`` registers one MjLab camera on a robot body (by default a
D435-like depth camera at the G1 head position on ``torso_link``). Several
observation terms (``hdmi.head_depth``, ``hdmi.object_pcd_camera``) read the
same sensor, so depth and segmentation are rendered once per control step.

Task YAML::

    sensors:
      head_camera:
        _target_: hdmi_head_camera
        body_name: torso_link
        pos: [0.0576, 0.0175, 0.4299]
        pitch_down_deg: 0.0
        fovy: 58.0
        width: 64
        height: 48
        data_types: [depth, segmentation]

Frame convention: a MuJoCo camera looks along its local ``-Z`` with ``+Y`` up.
The quaternion below makes the camera look along the parent body's ``+X``
(robot forward) with ``+Z`` up, then pitches it down by ``pitch_down_deg``
about the body's ``+Y`` axis. With ``pitch_down_deg=0`` this equals the MuJoCo
``xyaxes="0 -1 0 0 0 1"`` camera, i.e. quaternion (w, x, y, z) =
(0.5, 0.5, -0.5, -0.5).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from active_adaptation.registry import Registry


def head_camera_quat_wxyz(pitch_down_deg: float = 0.0) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) of a camera looking along body +X, pitched down."""
    import mujoco

    t = math.radians(float(pitch_down_deg))
    forward = np.array([math.cos(t), 0.0, -math.sin(t)])
    x_axis = np.array([0.0, -1.0, 0.0])  # image right = body -Y
    z_axis = -forward  # MuJoCo cameras look along -Z
    y_axis = np.cross(z_axis, x_axis)  # image up
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns = camera axes in body frame
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    return tuple(float(v) for v in quat)


def head_camera(
    backend: str,
    name: str,
    *,
    body_name: str = "torso_link",
    pos: Sequence[float] = (0.0576, 0.0175, 0.4299),
    pitch_down_deg: float = 0.0,
    fovy: float = 58.0,
    width: int = 64,
    height: int = 48,
    data_types: Sequence[str] = ("depth", "segmentation"),
    enabled_geom_groups: Sequence[int] = (0, 1, 2),
    use_textures: bool = False,
    use_shadows: bool = False,
):
    """Build a :class:`mjlab.sensor.CameraSensorCfg` mounted on ``robot/{body_name}``.

    ``pos`` is the camera origin in the parent body frame (metres); the default is
    the Unitree G1 D435 mount on ``torso_link``. ``pitch_down_deg`` tilts the optical
    axis below the body +X axis (the real G1 camera is tilted ~47.6 deg).
    """
    if backend != "mjlab":
        raise NotImplementedError("hdmi_head_camera currently supports only the MjLab backend")

    from mjlab.sensor import CameraSensorCfg

    return CameraSensorCfg(
        name=name,
        parent_body=f"robot/{body_name}",
        pos=tuple(float(v) for v in pos),
        quat=head_camera_quat_wxyz(pitch_down_deg),
        fovy=float(fovy),
        width=int(width),
        height=int(height),
        data_types=tuple(str(t) for t in data_types),
        use_textures=bool(use_textures),
        use_shadows=bool(use_shadows),
        enabled_geom_groups=tuple(int(g) for g in enabled_geom_groups),
    )


Registry.instance().register("sensor", "hdmi_head_camera", head_camera)

__all__ = ["head_camera", "head_camera_quat_wxyz"]
