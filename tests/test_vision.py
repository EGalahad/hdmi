"""Unit tests for the head-camera observation math and the depth encoder."""

from __future__ import annotations

import math

import pytest
import torch

from hdmi.sensors import head_camera_quat_wxyz
from hdmi.tasks.vision import (
    camera_intrinsics,
    pixel_directions,
    select_fixed_points,
    world_points_to_anchor_frame,
)
from hdmi_learning.common import DepthEncoder


def test_intrinsics_from_fovy():
    fx, fy, cx, cy = camera_intrinsics(64, 48, 58.0)
    assert fx == fy
    assert math.isclose(fy, 24.0 / math.tan(math.radians(29.0)), rel_tol=1e-9)
    assert (cx, cy) == (32.0, 24.0)


def test_unprojection_of_a_plane():
    dirs = pixel_directions(64, 48, 58.0)
    assert dirs.shape == (48, 64, 3)
    points = dirs * 1.0  # planar depth 1 m everywhere
    # camera looks along -Z
    assert torch.allclose(points[..., 2], torch.full((48, 64), -1.0))
    # horizontal extent = tan(hfov/2) with hfov from the 4:3 aspect ratio
    half_w = math.tan(math.radians(29.0)) * 64.0 / 48.0
    assert points[..., 0].max().item() < half_w and points[..., 0].max().item() > half_w * 0.95
    assert points[..., 0].min().item() > -half_w
    # row 0 is the top of the image (+Y), left column is -X
    assert (points[0, :, 1] > 0).all() and (points[-1, :, 1] < 0).all()
    assert (points[:, 0, 0] < 0).all() and (points[:, -1, 0] > 0).all()


def test_head_camera_quat_looks_forward():
    import mujoco

    quat = torch.tensor(head_camera_quat_wxyz(0.0))
    assert torch.allclose(quat.abs(), torch.tensor([0.5, 0.5, 0.5, 0.5]))
    mat = torch.zeros(9, dtype=torch.float64)
    mujoco.mju_quat2Mat(mat.numpy(), quat.to(torch.float64).numpy())
    rot = mat.reshape(3, 3)
    forward = -rot[:, 2]  # camera -Z in the body frame
    up = rot[:, 1]
    assert torch.allclose(forward, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(up, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6)

    quat = torch.tensor(head_camera_quat_wxyz(30.0))
    mujoco.mju_quat2Mat(mat.numpy(), quat.to(torch.float64).numpy())
    forward = -mat.reshape(3, 3)[:, 2]
    assert forward[2] < 0 and math.isclose(forward[0].item(), math.cos(math.radians(30.0)), abs_tol=1e-6)


def test_select_fixed_points_modes():
    torch.manual_seed(0)
    points = torch.arange(3 * 10 * 3, dtype=torch.float32).reshape(3, 10, 3)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[0] = True  # more than enough
    mask[1, :3] = True  # fewer than requested
    # row 2: nothing visible
    selected, count = select_fixed_points(points, mask, 5)
    assert selected.shape == (3, 5, 3)
    assert count.tolist() == [10, 3, 0]
    # row 0: five distinct valid points
    ids0 = (selected[0, :, 0] / 3).round().long()
    assert ids0.unique().numel() == 5
    # row 1: only valid points, cyclically repeated (first three distinct)
    ids1 = ((selected[1, :, 0] - 30) / 3).round().long()
    assert set(ids1.tolist()) <= {0, 1, 2} and ids1[:3].unique().numel() == 3
    # row 2: zeros
    assert torch.all(selected[2] == 0)


def test_world_points_to_anchor_frame_matches_yaw_only_transform():
    anchor_pos = torch.tensor([[1.0, 2.0, 0.7]])
    yaw = math.radians(90.0)
    anchor_quat = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
    points_w = torch.tensor([[[1.0, 3.0, 0.5]]])  # 1 m ahead of the anchor (+Y world = +X anchor)
    local = world_points_to_anchor_frame(points_w, anchor_pos, anchor_quat)
    assert torch.allclose(local, torch.tensor([[[1.0, 0.0, 0.5]]]), atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_depth_encoder_shapes(dtype):
    encoder = DepthEncoder(1, (16, 32, 64), (5, 3, 3), feature_dim=64)
    def make(*lead):
        if dtype == torch.uint8:
            return torch.randint(0, 256, (*lead, 1, 48, 64), dtype=torch.uint8)
        return torch.rand(*lead, 1, 48, 64)
    assert encoder(make(7)).shape == (7, 64)
    assert encoder(make(4, 3)).shape == (4, 3, 64)
    assert encoder(make()).shape == (64,)


def test_depth_encoder_onnx_export(tmp_path):
    encoder = DepthEncoder(1, (16, 32), (5, 3), feature_dim=32)
    encoder(torch.zeros(2, 1, 48, 64))  # materialise lazy layers
    encoder.eval()
    onnx = pytest.importorskip("onnx")
    del onnx
    path = tmp_path / "depth.onnx"
    torch.onnx.export(encoder, (torch.zeros(1, 48, 64, dtype=torch.uint8),), str(path), dynamo=True)
    assert path.exists()
