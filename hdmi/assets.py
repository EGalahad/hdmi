from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

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


def _make_object_spec(
    dataset_root: str | Path,
    *,
    name: str,
    mass: float,
    rgba: Sequence[float],
):
    import mujoco

    root = _resolve_dataset_root(dataset_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mjcf_path = root / manifest["mjcf"]
    combined_root = ET.parse(mjcf_path).getroot()
    object_body = combined_root.find(".//body[@name='object']")
    if object_body is None:
        raise ValueError(f"Dataset MJCF is missing body 'object': {mjcf_path}")
    object_geom = next(
        (geom for geom in object_body.findall("geom") if geom.get("mesh")), None
    )
    if object_geom is None:
        raise ValueError(f"Dataset object body has no mesh geom: {mjcf_path}")
    source_mesh_name = object_geom.attrib["mesh"]
    mesh = combined_root.find(f"./asset/mesh[@name='{source_mesh_name}']")
    if mesh is None:
        raise ValueError(f"Dataset MJCF is missing mesh {source_mesh_name!r}")
    compiler = combined_root.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_path = (mjcf_path.parent / mesh_dir / mesh.attrib["file"]).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    mjcf = ET.Element("mujoco", model=f"{name}_mesh")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "mesh",
        name="object_mesh",
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
        mesh="object_mesh",
        mass=str(float(mass)),
        rgba=" ".join(str(float(value)) for value in rgba),
        **{
            key: object_geom.attrib[key]
            for key in ("pos", "quat")
            if key in object_geom.attrib
        },
    )
    return mujoco.MjSpec.from_string(ET.tostring(mjcf, encoding="unicode"))


def make_object_mesh(
    backend: str,
    *,
    dataset_root: str | Path,
    name: str = "object",
    mass: float = 2.0,
    rgba: Sequence[float] = (0.4, 0.3, 0.2, 1.0),
):
    """Create one separate rigid runtime entity from an any4hdmi object mesh."""
    if backend != "mjlab":
        raise NotImplementedError(
            "The dataset object mesh asset currently supports only the MjLab backend"
        )

    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    def spec_fn():
        return _make_object_spec(
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


def make_suitcase_mesh(backend: str, **kwargs):
    """Backward-compatible suitcase asset entry point."""
    return make_object_mesh(backend, **kwargs)


Registry.instance().register("asset", "hdmi_object_mesh", make_object_mesh)
Registry.instance().register("asset", "hdmi_suitcase_mesh", make_suitcase_mesh)
