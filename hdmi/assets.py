from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from active_adaptation.registry import Registry


def _resolve_dataset_root(path: str | Path) -> Path:
    from any4hdmi.dataset.loading import find_any4hdmi_root, resolve_input_paths

    inputs = resolve_input_paths(Path(__file__).resolve().parents[3], path)
    if len(inputs) != 1:
        raise ValueError(f"Expected one suitcase dataset root, got {len(inputs)}")
    root = find_any4hdmi_root(inputs[0])
    if root is None:
        raise FileNotFoundError(f"Missing any4hdmi manifest above {inputs[0]}")
    return root


def _make_suitcase_spec(
    dataset_root: str | Path,
    *,
    name: str,
    mass: float,
    rgba: Sequence[float],
):
    import mujoco

    root = _resolve_dataset_root(dataset_root)
    combined_root = ET.parse(root / "g1-suitcase.xml").getroot()
    mesh = combined_root.find("./asset/mesh[@name='suitcase_mesh']")
    if mesh is None:
        raise ValueError("Dataset MJCF is missing asset mesh 'suitcase_mesh'")
    mesh_path = (root / "meshes" / mesh.attrib["file"]).absolute()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    mjcf = ET.Element("mujoco", model=f"{name}_mesh")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "mesh",
        name="suitcase_mesh",
        file=str(mesh_path),
        scale=mesh.attrib.get("scale", "1 1 1"),
    )
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name=name)
    ET.SubElement(body, "freejoint", name=f"{name}_root")
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_collision",
        type="mesh",
        mesh="suitcase_mesh",
        mass=str(float(mass)),
        rgba=" ".join(str(float(value)) for value in rgba),
    )
    return mujoco.MjSpec.from_string(ET.tostring(mjcf, encoding="unicode"))


def make_suitcase_mesh(
    backend: str,
    *,
    dataset_root: str | Path,
    name: str = "object",
    mass: float = 2.0,
    rgba: Sequence[float] = (0.4, 0.3, 0.2, 1.0),
):
    """Create the separate runtime entity from the dataset's suitcase mesh."""
    if backend != "mjlab":
        raise NotImplementedError(
            "The dataset suitcase mesh asset currently supports only the MjLab backend"
        )

    from mjlab.utils.spec_config import CollisionCfg

    from active_adaptation.assets.asset_cfg import EntityCfg

    def spec_fn():
        return _make_suitcase_spec(
            dataset_root,
            name=name,
            mass=mass,
            rgba=rgba,
        )

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(),
        spec_fn=spec_fn,
        articulation=None,
        collisions=(
            CollisionCfg(
                geom_names_expr=(f"{name}_collision",),
                contype=1,
                conaffinity=1,
                condim=3,
                priority=0,
                solref=(0.02, 1),
                friction=(1.0, 5e-3, 5e-4),
            ),
        ),
    )


Registry.instance().register("asset", "hdmi_suitcase_mesh", make_suitcase_mesh)


def _dataset_variant(
    path: str | Path,
    *,
    name: str,
    mass: float,
    rgba: Sequence[float],
    simplify_collision: bool = True,
    ballast_fraction: float = 0.0,
    up_direction: str = "+z",
    collision_vertex_count: int = 32,
    support_plane_offset: float = 0.003,
) -> tuple[Any, np.ndarray]:
    import mujoco
    import trimesh
    from any4hdmi.dataset.loading import find_any4hdmi_root, resolve_input_paths

    inputs = resolve_input_paths(Path(__file__).resolve().parents[3], path)
    if len(inputs) != 1:
        raise ValueError(f"Object variant {name!r} requires exactly one dataset path")
    root = find_any4hdmi_root(inputs[0])
    if root is None:
        raise FileNotFoundError(f"Missing any4hdmi manifest above {inputs[0]}")
    manifest = json.loads((root / "manifest.json").read_text())
    mjcf_path = root / manifest["mjcf"]
    combined = ET.parse(mjcf_path).getroot()
    body = combined.find(".//body[@name='object']")
    if body is None:
        raise ValueError(f"{mjcf_path} is missing body 'object'")
    source_geom = next(
        (geom for geom in body.findall("geom") if geom.get("mesh")), None
    )
    source_mesh_name = source_geom.get("mesh") if source_geom is not None else "object_mesh"
    source_mesh = combined.find(f"./asset/mesh[@name='{source_mesh_name}']")
    if source_mesh is None:
        raise ValueError(f"{mjcf_path} is missing object mesh asset")
    compiler = combined.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_path = (mjcf_path.parent / mesh_dir / source_mesh.get("file", "")).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    scale = np.fromstring(source_mesh.get("scale", "1 1 1"), sep=" ")
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.shape != (3,):
        raise ValueError(f"Invalid mesh scale in {mjcf_path}: {scale}")
    geom_pos = np.fromstring(
        source_geom.get("pos", "0 0 0") if source_geom is not None else "0 0 0",
        sep=" ",
    )
    geom_quat = np.fromstring(
        source_geom.get("quat", "1 0 0 0")
        if source_geom is not None
        else "1 0 0 0",
        sep=" ",
    )

    # HF snapshot paths are extensionless blobs; retain the source MJCF suffix
    # so trimesh can select the correct loader.
    mesh_suffix = Path(source_mesh.get("file", "")).suffix.lstrip(".") or None
    mesh = trimesh.load(
        mesh_path, file_type=mesh_suffix, force="mesh", process=False
    )
    vertices = np.asarray(mesh.vertices) * scale
    points, _ = trimesh.sample.sample_surface(mesh, 256, seed=0)
    points = points * scale
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, geom_quat)
    rotation = rotation.reshape(3, 3)
    vertices = vertices @ rotation.T + geom_pos
    points = points @ rotation.T + geom_pos
    try:
        up_axis, up_sign = {
            "+x": (0, 1.0),
            "-x": (0, -1.0),
            "+y": (1, 1.0),
            "-y": (1, -1.0),
            "+z": (2, 1.0),
            "-z": (2, -1.0),
        }[up_direction]
    except KeyError as error:
        raise ValueError(f"Invalid up direction {up_direction!r}") from error
    if simplify_collision:
        from scipy.spatial import ConvexHull

        if collision_vertex_count < 4:
            raise ValueError("collision_vertex_count must be at least 4")
        if support_plane_offset < 0:
            raise ValueError("support_plane_offset must be nonnegative")
        source_shape = trimesh.Trimesh(
            vertices=vertices, faces=mesh.faces, process=False
        )
        hull_vertices = np.asarray(source_shape.convex_hull.vertices)
        selected = [
            int(np.argmax(np.linalg.norm(hull_vertices - hull_vertices.mean(0), axis=1)))
        ]
        min_distance = np.linalg.norm(
            hull_vertices - hull_vertices[selected[0]], axis=1
        )
        for _ in range(1, min(collision_vertex_count, len(hull_vertices))):
            selected.append(int(np.argmax(min_distance)))
            min_distance = np.minimum(
                min_distance,
                np.linalg.norm(hull_vertices - hull_vertices[selected[-1]], axis=1),
            )
        reduced_vertices = hull_vertices[selected]
        reduced_hull = ConvexHull(reduced_vertices)
        simplified = trimesh.Trimesh(
            vertices=reduced_vertices,
            faces=reduced_hull.simplices,
            process=False,
        )
        simplified.fix_normals()
        simplified_vertices = np.asarray(simplified.vertices)
        bottom = (
            simplified_vertices[:, up_axis].min()
            if up_sign > 0
            else simplified_vertices[:, up_axis].max()
        )
        support_plane = bottom + up_sign * support_plane_offset
        kept = (simplified_vertices[:, up_axis] - support_plane) * up_sign >= 0
        clipped_points = [simplified_vertices[kept]]
        for start, end in simplified.edges_unique:
            if kept[start] == kept[end]:
                continue
            fraction = (support_plane - simplified_vertices[start, up_axis]) / (
                simplified_vertices[end, up_axis]
                - simplified_vertices[start, up_axis]
            )
            clipped_points.append(
                (
                    simplified_vertices[start]
                    + fraction
                    * (simplified_vertices[end] - simplified_vertices[start])
                )[None]
            )
        collision_vertices = np.unique(
            np.round(np.concatenate(clipped_points), decimals=12), axis=0
        )
        collision_hull = ConvexHull(collision_vertices)
        collision_mesh = trimesh.Trimesh(
            vertices=collision_vertices,
            faces=collision_hull.simplices,
            process=False,
        )
        collision_mesh.fix_normals()
    else:
        collision_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=mesh.faces,
            process=False,
        )

    collision_mass = float(mass) * (1.0 - ballast_fraction)
    ballast_mass = float(mass) * ballast_fraction
    ballast_pos = np.zeros(3)
    ballast_plane = (
        collision_mesh.vertices[:, up_axis].min()
        if up_sign > 0
        else collision_mesh.vertices[:, up_axis].max()
    )
    ballast_pos[up_axis] = ballast_plane + up_sign * 0.01

    def spec_fn():
        mjcf = ET.Element("mujoco", model=f"{name}_mesh")
        asset = ET.SubElement(mjcf, "asset")
        ET.SubElement(
            asset,
            "mesh",
            name="object_mesh",
            file=str(mesh_path),
            scale=" ".join(str(float(value)) for value in scale),
        )
        ET.SubElement(
            asset,
            "mesh",
            name="object_collision_mesh",
            vertex=" ".join(
                str(float(value)) for value in collision_mesh.vertices.reshape(-1)
            ),
            face=" ".join(
                str(int(value)) for value in collision_mesh.faces.reshape(-1)
            ),
        )
        worldbody = ET.SubElement(mjcf, "worldbody")
        runtime_body = ET.SubElement(worldbody, "body", name="object")
        ET.SubElement(runtime_body, "freejoint", name="object_root")
        ET.SubElement(
            runtime_body,
            "geom",
            name="object_visual",
            type="mesh",
            mesh="object_mesh",
            mass="0",
            pos=" ".join(str(float(value)) for value in geom_pos),
            quat=" ".join(str(float(value)) for value in geom_quat),
            rgba=" ".join(str(float(value)) for value in rgba),
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            runtime_body,
            "geom",
            name="object_collision",
            type="mesh",
            mesh="object_collision_mesh",
            mass=str(collision_mass),
            rgba="0 0 0 0",
            contype="1",
            conaffinity="1",
            condim="3",
            priority="0",
            solref="0.02 1",
            friction="1 0.005 0.0005",
        )
        ET.SubElement(
            runtime_body,
            "geom",
            name="object_ballast",
            type="sphere",
            size="0.01",
            pos=" ".join(str(float(value)) for value in ballast_pos),
            mass=str(ballast_mass),
            density="0",
            rgba="0 0 0 0",
            contype="0",
            conaffinity="0",
        )
        return mujoco.MjSpec.from_string(ET.tostring(mjcf, encoding="unicode"))

    return spec_fn, points.astype(np.float32)


def make_object_mesh_variants(
    backend: str,
    *,
    motion_cfgs: Mapping[str, Mapping[str, object]],
    masses: Mapping[str, float],
    seed: int,
    rgba: Sequence[float] = (0.4, 0.3, 0.2, 1.0),
    simplify_collision: bool = True,
    ballast_fractions: Mapping[str, float] | None = None,
    up_directions: Mapping[str, str] | None = None,
    collision_vertex_count: int = 32,
    support_plane_offset: float = 0.003,
):
    """Create fixed per-world rigid-object mesh variants from motion datasets."""
    if backend != "mjlab":
        # TODO: add IsaacLab per-env rigid-object asset variants.
        raise NotImplementedError("Object mesh variants currently require MjLab")
    from mjlab.entity.variants import VariantEntityCfg

    import active_adaptation as aa

    names = tuple(motion_cfgs)
    if not names:
        raise ValueError("motion_cfgs must contain at least one object dataset")
    if set(masses) != set(names):
        raise ValueError(
            "Object mass names must exactly match motion datasets: "
            f"datasets={sorted(names)}, masses={sorted(masses)}"
        )
    ballast_fractions = dict(ballast_fractions or {})
    if not set(ballast_fractions) <= set(names):
        raise ValueError("Ballast names must be a subset of motion datasets")
    if any(not 0.0 <= float(value) < 1.0 for value in ballast_fractions.values()):
        raise ValueError("Ballast fractions must be in [0, 1)")
    up_directions = dict(up_directions or {})
    if up_directions and set(up_directions) != set(names):
        raise ValueError("Up directions must exactly match motion datasets")
    variants = {}
    surface_points = {}
    weights = []
    for name, cfg in motion_cfgs.items():
        path = cfg.get("path")
        if not isinstance(path, (str, Path)):
            raise TypeError(f"motion_cfgs[{name!r}].path must be one path")
        weight = float(cfg.get("weight", 0.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"motion_cfgs[{name!r}].weight must be positive")
        variants[name], surface_points[name] = _dataset_variant(
            path,
            name=name,
            mass=float(masses[name]),
            rgba=rgba,
            simplify_collision=simplify_collision,
            ballast_fraction=float(ballast_fractions.get(name, 0.0)),
            up_direction=up_directions.get(name, "+z"),
            collision_vertex_count=collision_vertex_count,
            support_plane_offset=support_plane_offset,
        )
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

    return ObjectVariantCfg(
        variants=variants,
        assignment=assignment,
        surface_points=surface_points,
        variant_masses={name: float(masses[name]) for name in names},
    )


Registry.instance().register(
    "asset", "hdmi_object_mesh_variants", make_object_mesh_variants
)
