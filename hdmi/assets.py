from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from active_adaptation.registry import Registry


def _resolve_dataset_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3] / root
    root = root.resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing any4hdmi manifest under {root}")
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
    mesh_path = (root / "meshes" / mesh.attrib["file"]).resolve()
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

    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

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
