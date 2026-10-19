from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from any4hdmi.dataset.full import FullMotionDataset
from mimic_lite.tasks.deferred import DeferredObservation, DeferredReward, DeferredTermination

from active_adaptation.envs.utils import find_sensor_bodies
from .command import RobotObjectTracking


class ContactLabelStore:
    """Route contact sidecars by the runtime motion-path order, never by object ID."""

    def __init__(self, dataset, device):
        datasets = getattr(dataset, "datasets", [dataset])
        if not all(isinstance(d, FullMotionDataset) for d in datasets):
            raise ValueError("Contact sidecars currently require full_motion=true")
        chunks, lengths = [], []
        self.body_names = None
        for runtime in datasets:
            if runtime.sample_id_span != len(runtime.motion_paths):
                raise ValueError("Contact routes require one motion ID per motion path")
            root = Path(runtime.motion_paths[0]).parent.parent
            meta = json.loads((root / "contact_manifest.json").read_text())
            if meta["fps"] != 50:
                raise ValueError("Contact sidecar requires the original 50 Hz timeline")
            if self.body_names is None:
                self.body_names = list(meta["body_names"])
            elif self.body_names != meta["body_names"]:
                raise ValueError("Contact body names/order disagree between datasets")
            label_path = root / meta["labels_file"]
            if hashlib.sha256(label_path.read_bytes()).hexdigest() != meta["labels_sha256"]:
                raise ValueError(f"Contact sidecar checksum mismatch: {label_path}")
            data = np.load(label_path, mmap_mode="r", allow_pickle=False)
            if data.dtype != np.int8 or data.shape[1] != len(self.body_names):
                raise ValueError("Expected lossless int8 contact labels")
            source_lengths = np.asarray(meta["motion_lengths"], dtype=np.int64)
            offsets = np.concatenate([[0], np.cumsum(source_lengths)])
            lookup = {name: i for i, name in enumerate(meta["motion_files"])}
            selected_lengths = []
            for path in runtime.motion_paths:
                relative = Path(path).relative_to(root).as_posix()
                i = lookup[relative]
                chunks.append(np.array(data[offsets[i]:offsets[i + 1]], copy=True))
                selected_lengths.append(int(source_lengths[i]))
            if selected_lengths != runtime.lengths.cpu().tolist():
                raise ValueError("Contact timeline differs from runtime motion lengths")
            lengths.extend(selected_lengths)
        self.labels = torch.from_numpy(np.concatenate(chunks)).to(device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.starts = self.lengths.cumsum(0) - self.lengths
        if len(lengths) != len(dataset.motion_paths):
            raise ValueError("Contact/motion routes are not aligned")

    def get_slice(self, motion_ids, starts, steps):
        frame = starts[:, None] + steps[None, :]
        frame = frame.clamp_min(0).minimum(self.lengths[motion_ids, None] - 1)
        return self.labels[self.starts[motion_ids, None] + frame]


class ContactRobotObjectTracking(RobotObjectTracking, namespace="hdmi"):
    def __init__(self, *, contact_future_steps=(0, 5, 10, 15, 20), **kwargs):
        self.contact_store = None
        self._contact_future_steps_cfg = list(contact_future_steps)
        super().__init__(**kwargs)

    def _initialize(self, env):
        super()._initialize(env)
        if abs(env.step_dt - .02) > 1e-8:
            raise ValueError("Weave contacts must use the original 50 Hz policy clock")
        self.contact_store = ContactLabelStore(self.dataset, self.device)
        missing = set(self.contact_store.body_names) - set(self.asset.body_names)
        if missing:
            raise ValueError(f"Contact labels refer to missing robot bodies: {sorted(missing)}")
        self.contact_future_steps = torch.tensor(self._contact_future_steps_cfg, device=self.device)
        self._contact_reward_step = torch.tensor([1], device=self.device)
        self._refresh_future_buffers()

    def _refresh_future_buffers(self):
        super()._refresh_future_buffers()
        if self.contact_store is not None:
            self.future_contact_labels = self.contact_store.get_slice(
                self.motion_ids, self.obs_motion_t, self.contact_future_steps
            )
            # Rewards and termination compare the state after one policy frame,
            # matching RobotTracking.reward_current_step_index (future offset 1).
            self.ref_contact_labels = self.contact_store.get_slice(
                self.motion_ids, self.obs_motion_t, self._contact_reward_step
            )[:, 0]


class reference_contact_labels(DeferredObservation, namespace="hdmi"):
    def _initialize_impl(self):
        if not isinstance(self.command_manager, ContactRobotObjectTracking):
            raise ValueError("Reference contact observation requires contact-aware command")

    def compute(self):
        return self.command_manager.future_contact_labels.float().flatten(1)


def contact_match_score(labels: torch.Tensor, force: torch.Tensor, saturate_force: float):
    """Weave semantics: +1 expects contact, -1 forbids it, 0 is ignored."""
    mask = labels != 0
    strength = (force / saturate_force).clamp(0, 1)
    score = 1 - ((labels > 0).float() - strength).abs()
    return (score * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)


class HandObjectContact:
    def __init__(self, env, sensor_name, body_names):
        command = env.command_manager
        self.sensor = env.scene.sensors[sensor_name]
        indices, names = find_sensor_bodies(command.asset, self.sensor, body_names)
        if not names:
            raise ValueError("No hand bodies matched the object contact sensor")
        self.names = names
        self.sensor_indices = torch.tensor(indices, device=env.device)
        self.label_indices = torch.tensor(
            [command.contact_store.body_names.index(name) for name in names], device=env.device
        )
        self.command = command

    def labels(self):
        return self.command.ref_contact_labels[:, self.label_indices]

    def force_magnitude(self):
        data = self.sensor.data
        force = data.force
        if force is None or force.ndim != 3:
            raise ValueError("Expected filtered MjLab contact force (env, body, 3)")
        return force[:, self.sensor_indices].norm(dim=-1)


class contact_matching(DeferredReward, namespace="hdmi"):
    def _initialize_impl(self, sensor_name, body_names, saturate_force=5.0):
        if saturate_force <= 0:
            raise ValueError("saturate_force must be positive")
        self.contact = HandObjectContact(self.env, sensor_name, body_names)
        self.saturate_force = float(saturate_force)

    def _compute(self):
        return contact_match_score(
            self.contact.labels(), self.contact.force_magnitude(), self.saturate_force
        )


class lost_expected_contact(DeferredTermination, namespace="hdmi"):
    def _initialize_impl(self, sensor_name, body_names, max_lost_frames=10, threshold=1.0):
        if max_lost_frames < 1 or threshold < 0:
            raise ValueError("Invalid lost-contact thresholds")
        self.contact = HandObjectContact(self.env, sensor_name, body_names)
        self.max_lost_frames = int(max_lost_frames)
        self.threshold = float(threshold)
        self.lost_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids, reset_td=None):
        self.lost_counter[env_ids] = 0

    def update(self):
        expected = (self.contact.labels() > 0).any(-1)
        actual = (self.contact.force_magnitude() > self.threshold).any(-1)
        lost = expected & ~actual
        self.lost_counter = torch.where(lost, self.lost_counter + 1, 0)

    def compute(self, termination):
        return (self.lost_counter >= self.max_lost_frames).unsqueeze(-1)


class object_contact_flags(DeferredObservation, namespace="hdmi"):
    """Measured robot/object contact flags in reference-label body order."""

    def _initialize_impl(self, sensor_name, threshold=1.0):
        if threshold < 0:
            raise ValueError("Contact threshold must be nonnegative")
        command = self.command_manager
        if not isinstance(command, ContactRobotObjectTracking):
            raise ValueError("Object contacts require the contact-aware command")
        self.sensor = self.env.scene.sensors[sensor_name]
        indices, names = find_sensor_bodies(command.asset, self.sensor, ".*")
        lookup = dict(zip(names, indices))
        missing = set(command.contact_store.body_names) - set(lookup)
        if missing:
            raise ValueError(f"Contact sensor is missing reference bodies: {sorted(missing)}")
        self.sensor_indices = torch.tensor(
            [lookup[name] for name in command.contact_store.body_names],
            device=self.env.device,
        )
        self.threshold = float(threshold)

    def compute(self):
        force = self.sensor.data.force
        if force is None or force.ndim != 3 or force.shape[-1] != 3:
            raise ValueError("Expected filtered robot/object contact forces (env, body, 3)")
        return (force[:, self.sensor_indices].norm(dim=-1) > self.threshold).float()
