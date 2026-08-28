import torch

from hdmi.tasks.observations import _spatial_motion_from_local_poses


def _compose(
    position: torch.Tensor,
    rotation: torch.Tensor,
    offset_position: torch.Tensor,
    offset_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        position + torch.einsum("nhij,j->nhi", rotation, offset_position),
        rotation @ offset_rotation,
    )


def test_spatial_motion_is_object_frame_invariant_and_zero_step_is_identity():
    dtype = torch.float64
    angle = torch.tensor([0.0, 0.3, -0.4, 0.8], dtype=dtype)
    cos, sin = angle.cos(), angle.sin()
    rotation = torch.zeros(1, 4, 3, 3, dtype=dtype)
    rotation[0, :, 0, 0] = cos
    rotation[0, :, 0, 1] = -sin
    rotation[0, :, 1, 0] = sin
    rotation[0, :, 1, 1] = cos
    rotation[0, :, 2, 2] = 1.0
    position = torch.tensor(
        [[[0.2, -0.1, 0.4], [0.3, 0.0, 0.5], [0.4, 0.1, 0.6], [0.5, 0.2, 0.7]]],
        dtype=dtype,
    )

    offset_angle = torch.tensor(0.6, dtype=dtype)
    offset_rotation = torch.tensor(
        [
            [offset_angle.cos(), 0.0, offset_angle.sin()],
            [0.0, 1.0, 0.0],
            [-offset_angle.sin(), 0.0, offset_angle.cos()],
        ],
        dtype=dtype,
    )
    offset_position = torch.tensor([0.3, -0.2, 0.1], dtype=dtype)

    expected = _spatial_motion_from_local_poses(position, rotation, 1)
    reframed_pose = _compose(
        position, rotation, offset_position, offset_rotation
    )
    actual = _spatial_motion_from_local_poses(*reframed_pose, 1)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[0][:, 1], torch.zeros(1, 3, dtype=dtype))
    torch.testing.assert_close(actual[1][:, 1], torch.eye(3, dtype=dtype)[None])
