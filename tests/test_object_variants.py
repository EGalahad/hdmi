from pathlib import Path
import json
import mujoco
import numpy as np
import pytest
import trimesh
from hdmi.assets import make_object_mesh_variants
from mjlab.entity.variants import build_merged_variant_spec


def _dataset(root: Path, size: float, mass: float):
    root.mkdir()
    trimesh.creation.box((size, size, size)).export(root / 'visual.obj')
    (root / 'manifest.json').write_text(json.dumps({'format_version': 2, 'mjcf': 'object.xml'}))
    (root / 'object.xml').write_text(f'''<mujoco>
      <asset><mesh name="visual" file="visual.obj"/></asset>
      <worldbody><body name="object"><freejoint name="object_root"/>
      <geom name="object_visual" type="mesh" mesh="visual" mass="0" contype="0" conaffinity="0"/>
      <geom name="object_collision" type="mesh" mesh="visual" mass="{mass}" friction="0.8 0.004 0.0003"/>
      </body></worldbody></mujoco>''')
    return root


def test_prebuilt_variants_preserve_physics_points_and_assignment(tmp_path):
    first, second = _dataset(tmp_path / 'a', 1, 1), _dataset(tmp_path / 'b', 2, 4)
    kwargs = dict(backend='mjlab', motion_cfgs={
        'a': {'path': str(first), 'weight': 1},
        'b': {'path': str(second), 'weight': 3}}, seed=7)
    cfg, again = make_object_mesh_variants(**kwargs), make_object_mesh_variants(**kwargs)
    assert cfg.variant_masses == {'a': 1, 'b': 4}
    np.testing.assert_array_equal(cfg.assignment(8192), again.assignment(8192))
    np.testing.assert_allclose(np.bincount(cfg.assignment(8192))/8192, [.25,.75], atol=.02)
    for name, root, bound in [('a', first, .5), ('b', second, 1)]:
        model = cfg.variants[name]().compile()
        original = mujoco.MjModel.from_xml_path(str(root / 'object.xml'))
        for field in ['body_mass', 'body_inertia', 'body_ipos', 'geom_friction', 'geom_contype']:
            np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
        assert cfg.surface_points[name].shape == (256, 3)
        assert np.abs(cfg.surface_points[name]).max() <= bound + 1e-6
        np.testing.assert_array_equal(cfg.surface_points[name], again.surface_points[name])
    build_merged_variant_spec(cfg)


def test_missing_prebuilt_asset_fails_without_modifying_dataset(tmp_path):
    root = _dataset(tmp_path / 'a', 1, 1)
    (root / 'object.xml').unlink()
    with pytest.raises(FileNotFoundError, match='object.xml'):
        make_object_mesh_variants('mjlab', motion_cfgs={'a': {'path': str(root), 'weight': 1}}, seed=0)
    assert not (root / 'object.xml').exists()


def test_invalid_weight_rejected(tmp_path):
    root = _dataset(tmp_path / 'a', 1, 1)
    with pytest.raises(ValueError, match='positive'):
        make_object_mesh_variants('mjlab', motion_cfgs={'a': {'path': str(root), 'weight': 0}}, seed=0)


def test_subtree_split_keeps_one_root_and_physics(tmp_path):
    from hdmi.assets import _subtree_spec
    path = tmp_path / "combined.xml"
    path.write_text('<mujoco><asset/><asset><texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0 0 0" width="16" height="16"/></asset><worldbody>\n      <body name="pelvis"><freejoint/><geom name="robot_geom" type="sphere" size=".2" mass="3"/>\n        <body name="arm"><joint name="arm_joint"/><geom type="sphere" size=".1" mass="1"/></body>\n      </body>\n      <body name="object"><freejoint/><geom name="object_visual" type="box" size=".2 .3 .4" mass="2" friction=".7 .003 .0002"/></body>\n      </worldbody><actuator><motor joint="arm_joint"/></actuator></mujoco>')
    combined = mujoco.MjModel.from_xml_path(str(path))
    for root, nq in [("pelvis", 8), ("object", 7)]:
        model = _subtree_spec(path, root).compile()
        assert model.nq == nq
        assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
        for field in ["mass", "inertia", "ipos"]:
            np.testing.assert_allclose(getattr(model.body(root), field), getattr(combined.body(root), field))
    assert _subtree_spec(path, "object").compile().nu == 0
    assert len(_subtree_spec(path, "object").textures) == 0
