import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np

# Dataset preparation also works without importing the GPU task registry.
spec = importlib.util.spec_from_file_location("palm_dataset", Path(__file__).parents[1] / "hdmi/palm_dataset.py")
palm_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(palm_dataset)


def test_palm_fk_preserves_dynamics_and_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    xml = '<mujoco><worldbody>' + ''.join(
        f'<body name="{side}_wrist_yaw_link" pos="0 0 1"><joint name="{side}"/><geom size=".05" mass="1"/></body>'
        for side in ("left", "right")
    ) + '</worldbody></mujoco>'
    (source / "robot.xml").write_text(xml)
    (source / "manifest.json").write_text(json.dumps({"mjcf": "robot.xml"}))
    target = palm_dataset.materialize_palm_dataset(source, tmp_path / "cache")
    assert (source / "robot.xml").read_text() == xml
    assert palm_dataset.materialize_palm_dataset(source, tmp_path / "cache") == target
    old = mujoco.MjModel.from_xml_path(str(source / "robot.xml"))
    new = mujoco.MjModel.from_xml_path(str(target / "robot.xml"))
    assert (old.nq, old.nv, old.nu) == (new.nq, new.nv, new.nu)
    a, b = mujoco.MjData(old), mujoco.MjData(new)
    a.qpos[:] = b.qpos[:] = [.6, -.4]
    a.qvel[:] = b.qvel[:] = [.2, -.1]
    mujoco.mj_forward(old, a)
    mujoco.mj_forward(new, b)
    np.testing.assert_allclose(a.qacc, b.qacc)
    for side in ("left", "right"):
        wrist, palm = new.body(side + "_wrist_yaw_link").id, new.body(side + "_palm_link").id
        np.testing.assert_allclose(b.xpos[palm], b.xpos[wrist] + b.xmat[wrist].reshape(3, 3) @ [.1, 0, 0])
        np.testing.assert_allclose(b.xquat[palm], b.xquat[wrist])
