from types import SimpleNamespace
from unittest.mock import patch

import torch
from mimic_lite.tasks.command import RobotTracking
from hdmi.tasks.command import RobotObjectTracking


def test_play_starts_zero_and_training_preserves_sampler():
    ids = torch.tensor([0, 2])
    terminated = torch.tensor([True, False])
    command = SimpleNamespace(
        env=SimpleNamespace(training=False),
        play_from_start=True,
        t=torch.tensor([7, 8, 9]),
    )

    def sample(self, env_ids, *, terminated=None):
        self.t[env_ids] = 3

    # A real subclass instance is needed by super(); initialization requires a scene.
    obj = RobotObjectTracking.__new__(RobotObjectTracking)
    obj.__dict__.update(command.__dict__)
    with patch.object(
        RobotTracking, "_sample_motions", autospec=True, side_effect=sample
    ) as mock:
        RobotObjectTracking._sample_motions(obj, ids, terminated=terminated)
        assert obj.t.tolist() == [0, 8, 0]
        assert mock.call_args.kwargs["terminated"] is None
        obj.env.training = True
        RobotObjectTracking._sample_motions(obj, ids, terminated=terminated)
        assert obj.t.tolist() == [3, 8, 3]
        assert mock.call_args.kwargs["terminated"] is terminated
