"""
Synthetic Dataset Generator for Industrial Component Inspection.
Generates balanced datasets of normal and defective components with ground-truth masks.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import cv2
from ..hardware.frame_stream import SyntheticIndustrialComponentGenerator


def generate_offline_dataset(
    output_dir: str = "data/sample_dataset",
    num_samples_per_class: int = 20,
    width: int = 640,
    height: int = 480,
) -> Dict[str, Any]:
    """
    Generates an offline dataset of industrial parts organized by defect category.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generator = SyntheticIndustrialComponentGenerator(width, height)
    categories = [
        "normal",
        "surface_scratch",
        "structural_crack",
        "void_porosity",
        "corrosion_spot",
        "dimensional_misalignment",
    ]

    manifest: List[Dict[str, Any]] = []

    for cat in categories:
        cat_dir = out_path / cat
        cat_dir.mkdir(exist_ok=True)

        for i in range(num_samples_per_class):
            inject = (cat != "normal")
            frame, meta = generator.generate_frame(
                inject_defect=inject,
                defect_type=cat if inject else None,
                conveyor_offset=float(i * 0.15)
            )

            file_name = f"{cat}_{i:04d}.jpg"
            file_path = cat_dir / file_name
            cv2.imwrite(str(file_path), frame)

            record = {
                "file_name": file_name,
                "relative_path": f"{cat}/{file_name}",
                "category": cat,
                "is_defect": inject,
                "component_type": meta.get("component_type"),
            }
            manifest.append(record)

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dataset_path": str(out_path),
        "total_images": len(manifest),
        "categories": categories,
        "manifest_path": str(manifest_path),
    }
