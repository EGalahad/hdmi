from pathlib import Path

import mujoco
import numpy as np

from hdmi.assets import _make_suitcase_spec


def test_runtime_suitcase_matches_dataset_mesh_and_frame():
    root = Path(__file__).resolve().parents[4] / "any4hdmi/output/g1/omomo_suitcase_gmr_accepted"
    runtime = _make_suitcase_spec(
        root, name="object", mass=2.0, rgba=(0.4, 0.3, 0.2, 1.0)
    ).compile()
    combined = mujoco.MjModel.from_xml_path(str(root / "g1-suitcase.xml"))
    qpos = np.load(sorted((root / "motions").glob("*.npz"))[0])["qpos"]
    runtime_data, combined_data = mujoco.MjData(runtime), mujoco.MjData(combined)
    runtime_body = mujoco.mj_name2id(runtime, mujoco.mjtObj.mjOBJ_BODY, "object")
    combined_body = mujoco.mj_name2id(combined, mujoco.mjtObj.mjOBJ_BODY, "object")
    runtime_geom = mujoco.mj_name2id(
        runtime, mujoco.mjtObj.mjOBJ_GEOM, "object_collision"
    )
    combined_geom = mujoco.mj_name2id(
        combined, mujoco.mjtObj.mjOBJ_GEOM, "object_collision"
    )

    for frame in np.linspace(0, len(qpos) - 1, 10, dtype=int):
        runtime_data.qpos[:] = qpos[frame, 36:43]
        combined_data.qpos[:] = qpos[frame]
        mujoco.mj_forward(runtime, runtime_data)
        mujoco.mj_forward(combined, combined_data)
        np.testing.assert_array_equal(
            runtime_data.xpos[runtime_body], combined_data.xpos[combined_body]
        )
        np.testing.assert_array_equal(
            runtime_data.xmat[runtime_body], combined_data.xmat[combined_body]
        )
        np.testing.assert_array_equal(
            runtime_data.geom_xpos[runtime_geom], combined_data.geom_xpos[combined_geom]
        )
        np.testing.assert_array_equal(
            runtime_data.geom_xmat[runtime_geom], combined_data.geom_xmat[combined_geom]
        )

    assert runtime.nmesh == 1
    np.testing.assert_allclose(
        runtime.mesh_vert,
        combined.mesh_vert[-runtime.mesh_vertnum[0] :],
        atol=1e-7,
    )
