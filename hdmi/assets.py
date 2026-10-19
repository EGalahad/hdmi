from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from active_adaptation.registry import Registry


def _resolve_dataset_root(path: str | Path) -> Path:
    from any4hdmi.dataset.loading import find_any4hdmi_root, resolve_input_paths
    inputs = resolve_input_paths(Path(__file__).resolve().parents[3], path)
    if len(inputs) != 1:
        raise ValueError(f"Expected one object dataset root, got {len(inputs)}")
    root = find_any4hdmi_root(inputs[0])
    if root is None:
        raise FileNotFoundError(f"Missing any4hdmi manifest above {inputs[0]}")
    return root


def _dataset_xml(dataset_root: str | Path) -> Path:
    root = _resolve_dataset_root(dataset_root)
    path = root / json.loads((root / "manifest.json").read_text())["mjcf"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _subtree_spec(path: Path, body_name: str):
    """Select one floating root without changing its physical properties."""
    import mujoco
    xml = ET.parse(path).getroot()
    world = xml.find("worldbody")
    chosen = world.find(f"body[@name='{body_name}']")
    if chosen is None:
        raise ValueError(f"{path} has no top-level body {body_name!r}")
    for node in list(world):
        if node is not chosen:
            world.remove(node)
    # Combined datasets store actuator/sensor/contact definitions for the robot.
    if body_name == "object":
        for section in ("actuator", "sensor", "contact", "equality", "tendon"):
            node = xml.find(section)
            if node is not None:
                xml.remove(node)
    keyframe = xml.find("keyframe")
    if keyframe is not None:
        xml.remove(keyframe)  # EntityCfg supplies the selected entity's initial state.
    compiler = xml.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    used_meshes = {geom.get("mesh") for geom in chosen.iter("geom")}
    assets = xml.find("asset")
    for extra in xml.findall("asset")[1:]:
        assets.extend(list(extra))
        xml.remove(extra)
    if assets is not None:
        for mesh in list(assets.findall("mesh")):
            if mesh.get("name") not in used_meshes:
                assets.remove(mesh)
            elif mesh.get("file"):
                mesh.set("file", str((path.parent / mesh_dir / mesh.get("file")).absolute()))
    if assets is not None:
        used_materials = {g.get("material") for g in chosen.iter("geom")}
        used_materials.update(g.get("material") for g in xml.findall("./default//geom"))
        for material in list(assets.findall("material")):
            if material.get("name") not in used_materials:
                assets.remove(material)
        used_textures = {m.get("texture") for m in assets.findall("material")}
        for texture in list(assets.findall("texture")):
            if texture.get("name") not in used_textures or not texture.get("name"):
                assets.remove(texture)
    if compiler is not None:
        compiler.set("meshdir", "")
    spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
    if sum(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints) != 1:
        raise ValueError(f"{body_name!r} must contain exactly one free joint")
    return spec


def make_object_asset(backend: str, *, dataset_root: str | Path):
    if backend != "mjlab":
        raise NotImplementedError("Object assets currently require MjLab")
    from active_adaptation.assets.asset_cfg import EntityCfg
    root = _resolve_dataset_root(dataset_root)
    path = _dataset_xml(root)
    metadata = json.loads((root / "manifest.json").read_text())

    @dataclass
    class ObjectCfg(EntityCfg):
        object_name: str = "object"

    return ObjectCfg(init_state=ObjectCfg.InitialStateCfg(),
                     object_name=str(metadata.get("object_name", root.name)),
                     spec_fn=lambda: _subtree_spec(path, "object"), articulation=None)


def make_robot_asset(backend: str, *, dataset_root: str | Path, mode: int = 15):
    if backend != "mjlab":
        raise NotImplementedError("Combined robot assets currently require MjLab")
    from mimic_lite.assets.g1 import _build_g1_cfg
    asset = _build_g1_cfg(mode, backend)
    path = _dataset_xml(dataset_root)
    asset.config.spec_fn = lambda: _subtree_spec(path, "pelvis")
    return asset


Registry.instance().register("asset", "hdmi_object_asset", make_object_asset)
Registry.instance().register("asset", "hdmi_robot_from_motion", make_robot_asset)


def make_weave_robot_asset(backend: str, *, dataset_root: str | Path, mode: int = 15):
    """Use the converted G1+Inspire model with the existing G1 body controller."""
    if backend != "mjlab":
        raise NotImplementedError("Weave combined assets currently require MjLab")
    import copy
    import mujoco
    from mjlab.actuator import BuiltinPositionActuatorCfg

    asset = make_robot_asset(backend, dataset_root=dataset_root, mode=mode)
    model = asset.config.spec_fn().compile()
    joints = [model.joint(i).name for i in range(model.njnt)
              if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if len(joints) != 53:
        raise ValueError(f"G1+Inspire requires 53 movable joints, got {len(joints)}")
    asset.config.joint_names_simulation = joints
    asset.config.body_names_simulation = [model.body(i).name for i in range(1, model.nbody)]
    joint_mapping = copy.deepcopy(asset.config.joint_symmetry_mapping)
    body_mapping = copy.deepcopy(asset.config.spatial_symmetry_mapping)
    for name in joints:
        if name.startswith(("L_", "R_")):
            other = ("R_" if name.startswith("L_") else "L_") + name[2:]
            joint_mapping[name] = (1, other)
    for name in asset.config.body_names_simulation:
        if name.startswith(("L_", "R_")):
            body_mapping[name] = ("R_" if name.startswith("L_") else "L_") + name[2:]
        elif name not in body_mapping:
            body_mapping[name] = name
    asset.config.joint_symmetry_mapping = joint_mapping
    asset.config.spatial_symmetry_mapping = body_mapping
    asset.config.articulation.actuators = (*asset.config.articulation.actuators,
        BuiltinPositionActuatorCfg(
            target_names_expr=("[LR]_.*_joint",), effort_limit=1.0,
            stiffness=5.0, damping=0.5, armature=0.01, frictionloss=0.0,
        ))
    return asset


Registry.instance().register("asset", "hdmi_weave_robot_from_motion", make_weave_robot_asset)


def _surface_points(path: Path) -> np.ndarray:
    """Sample the visual mesh online, in the object's body coordinate frame."""
    import mujoco
    import trimesh
    xml = ET.parse(path).getroot()
    geom = xml.find("./worldbody/body[@name='object']/geom[@name='object_visual']")
    if geom is None:
        raise ValueError(f"{path} is missing object_visual")
    mesh = xml.find(f"./asset/mesh[@name='{geom.get('mesh')}']")
    if mesh is None:
        raise ValueError(f"{path} is missing the visual mesh asset")
    compiler = xml.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_file = mesh.attrib["file"]
    source = trimesh.load(path.parent / mesh_dir / mesh_file,
                          file_type=Path(mesh_file).suffix.lstrip("."),
                          force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(source, 256, seed=0)
    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
    quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return ((points * scale) @ rotation.reshape(3, 3).T + pos).astype(np.float32)


def make_object_mesh_variants(backend: str, *,
                              motion_cfgs: Mapping[str, Mapping[str, object]],
                              seed: int,
                              initial_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Load prebuilt variants; assign objects and sample visual points online."""
    if backend != "mjlab":
        raise NotImplementedError("Object mesh variants currently require MjLab")
    import mujoco
    from mjlab.entity.variants import VariantEntityCfg
    import active_adaptation as aa
    names = tuple(motion_cfgs)
    if not names:
        raise ValueError("motion_cfgs must contain at least one object dataset")
    variants, surface_points, masses, object_names, weights = {}, {}, {}, {}, []
    for name, cfg in motion_cfgs.items():
        path = cfg.get("path")
        if not isinstance(path, (str, Path)):
            raise TypeError(f"motion_cfgs[{name!r}].path must be one path")
        weight = float(cfg.get("weight", 0.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"motion_cfgs[{name!r}].weight must be positive")
        xml = _dataset_xml(path)
        metadata = json.loads((_resolve_dataset_root(path) / "manifest.json").read_text())
        object_names[name] = str(metadata.get("object_name", name))
        variants[name] = lambda xml=xml: _subtree_spec(xml, "object")
        surface_points[name] = _surface_points(xml)
        masses[name] = float(variants[name]().compile().body("object").mass[0])
        weights.append(weight)
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()

    def assignment(num_envs: int) -> np.ndarray:
        return np.random.default_rng(int(seed) + aa.get_local_rank()).choice(
            len(names), size=int(num_envs), p=probabilities
        )

    @dataclass
    class ObjectVariantCfg(VariantEntityCfg):
        surface_points: dict[str, np.ndarray] = field(default_factory=dict)
        variant_masses: dict[str, float] = field(default_factory=dict)
        object_names: dict[str, str] = field(default_factory=dict)

    return ObjectVariantCfg(
        init_state=ObjectVariantCfg.InitialStateCfg(pos=tuple(initial_pos)),
        variants=variants,
        assignment=assignment,
        surface_points=surface_points,
        variant_masses={name: float(masses[name]) for name in names},
        object_names=object_names,
    )


Registry.instance().register(
    "asset", "hdmi_object_mesh_variants", make_object_mesh_variants
)
