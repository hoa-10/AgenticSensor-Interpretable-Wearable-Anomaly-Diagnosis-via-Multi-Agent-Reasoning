from __future__ import annotations

import configparser
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "perception_layer" / "perception.config"
POSITIONS = ("hand", "chest", "ankle")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_config(config_path: str | Path) -> configparser.ConfigParser:
    path = resolve_project_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    return config


def get_config_path(
    config: configparser.ConfigParser,
    key: str,
    *,
    section: str = "paths",
    required: bool = True,
) -> Path | None:
    value = config.get(section, key, fallback="").strip()
    if not value:
        if required:
            raise ValueError(f"Missing [{section}] {key} in config.")
        return None
    return resolve_project_path(value)


def get_sensor_paths(config: configparser.ConfigParser) -> dict[str, Path]:
    return {
        position: get_config_path(config, f"{position}_path")
        for position in POSITIONS
    }


def infer_batch_id_from_sensor_paths(sensor_paths: dict[str, Path]) -> int:
    for path in sensor_paths.values():
        match = re.search(r"_batch_(\d+)\.parquet$", path.name)
        if match:
            return int(match.group(1))
    raise ValueError("Could not infer batch id from configured sensor paths.")


def infer_image_path_from_sensor_paths(sensor_paths: dict[str, Path]) -> Path:
    batch_id = infer_batch_id_from_sensor_paths(sensor_paths)
    first_path = next(iter(sensor_paths.values()))
    batch_dir = first_path.parent.parent
    return batch_dir / "image" / f"fused_batch_{batch_id:06d}.png"


def ensure_fused_image_from_sensor_paths(
    sensor_paths: dict[str, Path],
    sampling_rate: int = 50,
) -> Path:
    image_path = infer_image_path_from_sensor_paths(sensor_paths)
    if image_path.exists():
        return image_path

    from sensor_layer.temporal_fusion import TemporalFusion

    batch_id = infer_batch_id_from_sensor_paths(sensor_paths)
    batch_dir = next(iter(sensor_paths.values())).parent.parent
    fusion = TemporalFusion(
        data_dir=batch_dir,
        output_dir=batch_dir,
        sampling_rate=sampling_rate,
    )
    return fusion.plot_batch_id(
        batch_id=batch_id,
        batch_dir=batch_dir,
        image_dir=image_path.parent,
    )


def format_sensor_paths(sensor_paths: dict[str, Path]) -> str:
    return "\n".join(
        f"{position}_path: {sensor_paths[position]}"
        for position in POSITIONS
    )
