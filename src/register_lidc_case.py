"""Register one explicitly selected LIDC XML nodule annotation to its DICOM CT.

The script uses the annotation's SOP Instance UIDs to bind XML contours to one
exact DICOM series. It creates a checkpoint-compatible 32^3 HU-normalized ROI
and registration metadata. It does not infer a patient-to-robot calibration.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pydicom


ROI_SIZE = 32
HU_MIN = -1000.0
HU_MAX = 400.0


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> Optional[str]:
    for child in element:
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


def _descendant_text(element: ET.Element, name: str) -> Optional[str]:
    for child in element.iter():
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


def read_annotation_xml(path: Path) -> tuple[ET.Element, Optional[str], Optional[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"LIDC annotation XML not found: {path}")
    root = ET.parse(path).getroot()
    series_uid = _descendant_text(root, "SeriesInstanceUid")
    study_uid = _descendant_text(root, "StudyInstanceUID")
    return root, series_uid, study_uid


def annotation_records(root: ET.Element) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for session in (node for node in root.iter() if _local_name(node.tag) == "readingSession"):
        reader_id = _child_text(session, "servicingRadiologistID")
        for node in session:
            if _local_name(node.tag) != "unblindedReadNodule":
                continue
            nodule_id = _child_text(node, "noduleID")
            if not nodule_id:
                continue
            rois = [child for child in node if _local_name(child.tag) == "roi"]
            included_rois = [
                roi
                for roi in rois
                if (_child_text(roi, "inclusion") or "TRUE").upper() == "TRUE"
            ]
            sops = {
                _child_text(roi, "imageSOP_UID")
                for roi in included_rois
                if _child_text(roi, "imageSOP_UID")
            }
            malignancy_text = _descendant_text(node, "malignancy")
            records.append(
                {
                    "nodule_id": nodule_id,
                    "reader_id": reader_id,
                    "roi_count": len(included_rois),
                    "sop_instance_uids": sorted(sops),
                    "malignancy": int(malignancy_text) if malignancy_text and malignancy_text.isdigit() else None,
                    "element": node,
                }
            )
    return records


def select_annotation(
    records: list[dict[str, Any]], nodule_id: str, reader_id: Optional[str] = None
) -> dict[str, Any]:
    matches = [record for record in records if record["nodule_id"] == nodule_id]
    if reader_id is not None:
        matches = [record for record in matches if record["reader_id"] == reader_id]
    if not matches:
        available = ", ".join(
            f'{record["nodule_id"]} (reader {record["reader_id"] or "unknown"})'
            for record in records
        )
        raise ValueError(
            f"No matching annotation for nodule_id={nodule_id!r}"
            + (f", reader_id={reader_id!r}" if reader_id is not None else "")
            + (f". Available: {available}" if available else ". XML has no unblinded nodule annotations.")
        )
    if len(matches) != 1:
        readers = ", ".join(str(record["reader_id"]) for record in matches)
        raise ValueError(
            f"Nodule ID {nodule_id!r} appears in {len(matches)} reading sessions. "
            f"Specify --reader-id; matching readers: {readers}"
        )
    if matches[0]["roi_count"] == 0 or not matches[0]["sop_instance_uids"]:
        raise ValueError(f"Annotation {nodule_id!r} has no included image contours")
    return matches[0]


def _dicom_header(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception:
        return None


def load_dicom_volume(series_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one DICOM series and return HU volume, index-to-LPS affine, metadata."""
    if not series_dir.is_dir():
        raise NotADirectoryError(f"DICOM series directory not found: {series_dir}")

    entries: list[tuple[Path, Any]] = []
    for path in series_dir.rglob("*"):
        if not path.is_file():
            continue
        header = _dicom_header(path)
        if (
            header is not None
            and getattr(header, "SeriesInstanceUID", None)
            and str(getattr(header, "Modality", "CT")) == "CT"
        ):
            entries.append((path, header))
    if not entries:
        raise FileNotFoundError(f"No readable DICOM instances found under {series_dir}")

    series_uids = {str(header.SeriesInstanceUID) for _, header in entries}
    if len(series_uids) != 1:
        raise ValueError(
            "The supplied directory contains multiple DICOM SeriesInstanceUIDs; "
            "pass the exact series directory. Found: " + ", ".join(sorted(series_uids))
        )
    study_uids = {str(header.StudyInstanceUID) for _, header in entries if hasattr(header, "StudyInstanceUID")}
    if len(study_uids) > 1:
        raise ValueError("The supplied DICOM series contains multiple StudyInstanceUIDs")

    first = entries[0][1]
    required = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns")
    missing = [name for name in required if not hasattr(first, name)]
    if missing:
        raise ValueError("DICOM series is missing spatial metadata: " + ", ".join(missing))
    row_direction = np.asarray(first.ImageOrientationPatient[:3], dtype=float)
    column_direction = np.asarray(first.ImageOrientationPatient[3:], dtype=float)
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)
    if (
        not np.isfinite(row_direction).all()
        or not np.isfinite(column_direction).all()
        or not np.isfinite(pixel_spacing).all()
        or np.any(pixel_spacing <= 0)
        or not np.isclose(np.linalg.norm(row_direction), 1.0, atol=1e-3)
        or not np.isclose(np.linalg.norm(column_direction), 1.0, atol=1e-3)
        or not np.isclose(row_direction @ column_direction, 0.0, atol=1e-3)
    ):
        raise ValueError("Invalid DICOM orientation or pixel spacing metadata")
    normal = np.cross(row_direction, column_direction)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 0.99:
        raise ValueError("Invalid DICOM ImageOrientationPatient direction cosines")
    normal /= np.linalg.norm(normal)

    positions = [np.asarray(header.ImagePositionPatient, dtype=float) for _, header in entries]
    projections = np.asarray([position @ normal for position in positions])
    order = np.argsort(projections)
    entries = [entries[int(index)] for index in order]
    positions = [positions[int(index)] for index in order]
    projections = projections[order]
    if len(entries) < 2:
        raise ValueError("At least two CT slices are required to build a 3D ROI")
    if np.any(np.diff(projections) <= 1e-4):
        raise ValueError("DICOM series has duplicate or unordered slice positions")

    headers = [header for _, header in entries]
    rows, columns = int(first.Rows), int(first.Columns)
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    spacing = pixel_spacing
    for header in headers:
        if int(header.Rows) != rows or int(header.Columns) != columns:
            raise ValueError("DICOM slice matrix dimensions are inconsistent")
        if not np.allclose(header.ImageOrientationPatient, orientation, atol=1e-5):
            raise ValueError("DICOM slice orientations differ; cannot create a regular volume")
        if not np.allclose(header.PixelSpacing, spacing, atol=1e-5):
            raise ValueError("DICOM pixel spacing differs between slices")

    step_vectors = np.diff(np.stack(positions), axis=0)
    slice_step = np.median(step_vectors, axis=0)
    if not np.allclose(step_vectors, slice_step, atol=0.25):
        raise ValueError("DICOM slice spacing/positions are irregular; refusing inaccurate ROI mapping")

    sop_to_index: dict[str, int] = {}
    slices = []
    for index, (path, header) in enumerate(entries):
        sop_uid = str(getattr(header, "SOPInstanceUID", ""))
        if not sop_uid or sop_uid in sop_to_index:
            raise ValueError(f"DICOM series contains a missing or duplicate SOPInstanceUID: {sop_uid!r}")
        sop_to_index[sop_uid] = index
        dataset = pydicom.dcmread(str(path), force=False)
        pixels = dataset.pixel_array.astype(np.float32)
        pixels *= float(getattr(dataset, "RescaleSlope", 1.0))
        pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
        slices.append(pixels)

    volume = np.stack(slices, axis=0)
    x_direction = row_direction * float(spacing[1])
    y_direction = column_direction * float(spacing[0])
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = np.column_stack((x_direction, y_direction, slice_step))
    affine[:3, 3] = positions[0]
    metadata = {
        "series_instance_uid": next(iter(series_uids)),
        "study_instance_uid": next(iter(study_uids)) if study_uids else None,
        "sop_to_index": sop_to_index,
        "headers_by_sop": {str(header.SOPInstanceUID): header for header in headers},
        "slice_spacing_mm": float(np.median(np.diff(projections))),
    }
    return volume, affine, metadata


def annotation_center_world(
    annotation: dict[str, Any],
    headers_by_sop: dict[str, Any],
    expected_series_uid: str,
    image_shape_zyx: tuple[int, int, int],
    z_tolerance_mm: float,
) -> tuple[np.ndarray, list[str], float]:
    """Average per-slice contour centroids after mapping pixels through DICOM geometry."""
    element = annotation["element"]
    centers_lps: list[np.ndarray] = []
    used_sops: list[str] = []
    z_deltas: list[float] = []
    depth, height, width = image_shape_zyx

    for roi in (node for node in element if _local_name(node.tag) == "roi"):
        inclusion = (_child_text(roi, "inclusion") or "TRUE").upper()
        if inclusion != "TRUE":
            continue
        sop_uid = _child_text(roi, "imageSOP_UID")
        if not sop_uid:
            raise ValueError(f"Annotation {annotation['nodule_id']!r} contains a contour without imageSOP_UID")
        header = headers_by_sop.get(sop_uid)
        if header is None:
            raise ValueError(
                f"Annotation SOP Instance UID {sop_uid} is not present in the supplied DICOM series"
            )
        if str(header.SeriesInstanceUID) != expected_series_uid:
            raise ValueError(f"SOP Instance UID {sop_uid} belongs to a different DICOM series")
        if int(header.Rows) != height or int(header.Columns) != width:
            raise ValueError(f"DICOM image dimensions disagree at SOP Instance UID {sop_uid}")

        points: list[tuple[float, float]] = []
        for edge in (node for node in roi if _local_name(node.tag) == "edgeMap"):
            x_text = _child_text(edge, "xCoord")
            y_text = _child_text(edge, "yCoord")
            if x_text is None or y_text is None:
                continue
            x, y = float(x_text), float(y_text)
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"XML contour coordinate ({x}, {y}) is outside DICOM image dimensions "
                    f"({width}, {height}); check pixel-index convention"
                )
            points.append((x, y))
        if not points:
            continue

        x_center, y_center = np.mean(np.asarray(points, dtype=float), axis=0)
        image_position = np.asarray(header.ImagePositionPatient, dtype=float)
        iop = np.asarray(header.ImageOrientationPatient, dtype=float)
        pixel_spacing = np.asarray(header.PixelSpacing, dtype=float)
        world = (
            image_position
            + iop[:3] * x_center * pixel_spacing[1]
            + iop[3:] * y_center * pixel_spacing[0]
        )
        xml_z = _child_text(roi, "imageZposition")
        if xml_z is not None:
            delta = abs(float(header.ImagePositionPatient[2]) - float(xml_z))
            z_deltas.append(delta)
            if delta > z_tolerance_mm:
                raise ValueError(
                    f"XML imageZposition differs from its SOP-linked DICOM slice by {delta:.3f} mm "
                    f"(allowed {z_tolerance_mm:.3f} mm)"
                )
        centers_lps.append(world)
        used_sops.append(sop_uid)

    if not centers_lps:
        raise ValueError(f"Annotation {annotation['nodule_id']!r} has no usable included contours")
    return np.mean(np.stack(centers_lps), axis=0), sorted(set(used_sops)), max(z_deltas, default=0.0)


def extract_normalized_roi(
    volume_hu: np.ndarray, center_xyz: np.ndarray, roi_size: int = ROI_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    if roi_size <= 0 or roi_size % 2:
        raise ValueError("ROI size must be a positive even number")
    center = np.rint(np.asarray(center_xyz, dtype=float)).astype(int)
    if center.shape != (3,):
        raise ValueError("ROI center must be [x, y, z]")
    x, y, z = center.tolist()
    half = roi_size // 2
    depth, height, width = volume_hu.shape
    if not (half <= x <= width - half and half <= y <= height - half and half <= z <= depth - half):
        raise ValueError(
            f"32^3 ROI centered at voxel {center.tolist()} exceeds CT bounds {volume_hu.shape[::-1]}"
        )
    cube_hu = volume_hu[z-half:z+half, y-half:y+half, x-half:x+half]
    if cube_hu.shape != (roi_size, roi_size, roi_size):
        raise RuntimeError(f"Unexpected cropped ROI shape: {cube_hu.shape}")
    normalized = (np.clip(cube_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return normalized.astype(np.float32), cube_hu


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_preview(path: Path, cube_hu: np.ndarray, center_float_xyz: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center = np.asarray(center_float_xyz, dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    views = (
        (cube_hu[16, :, :], center[0] - (round(center[0]) - 16), center[1] - (round(center[1]) - 16), "Axial"),
        (cube_hu[:, 16, :], center[0] - (round(center[0]) - 16), center[2] - (round(center[2]) - 16), "Coronal"),
        (cube_hu[:, :, 16], center[2] - (round(center[2]) - 16), center[1] - (round(center[1]) - 16), "Sagittal"),
    )
    for axis, (image, marker_x, marker_y, title) in zip(axes, views):
        axis.imshow(image, cmap="gray", vmin=HU_MIN, vmax=HU_MAX, origin="lower")
        axis.scatter([marker_x], [marker_y], s=28, c="#e4572e", marker="+")
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle("LIDC XML registered ROI center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _repo_relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _prepare_config_update(path: Path, case_entry: dict[str, Any], replace_case: bool):
    if not path.is_file():
        raise FileNotFoundError(f"Cases config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("cases"), list):
        raise ValueError("Cases config must be a JSON object containing a cases array")
    matching = [item for item in config["cases"] if item.get("case_id") == case_entry["case_id"]]
    if matching and not replace_case:
        raise ValueError(
            f"{case_entry['case_id']} is already in {path}; pass --replace-case to replace its entry"
        )
    updated_cases = []
    for item in config["cases"]:
        if item.get("case_id") == case_entry["case_id"]:
            preserved = dict(item)
            preserved.update(case_entry)
            updated_cases.append(preserved)
        else:
            updated_cases.append(item)
    if not matching:
        updated_cases.append(case_entry)
    config["cases"] = updated_cases
    return config


def register_case(args: argparse.Namespace) -> dict[str, Any]:
    case_id = args.case_id
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("case_id may contain only letters, digits, underscores, and hyphens")

    xml_path = args.annotation_xml.expanduser().resolve()
    series_dir = args.ct_series.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    root, xml_series_uid, xml_study_uid = read_annotation_xml(xml_path)
    records = annotation_records(root)
    annotation = select_annotation(records, args.nodule_id, args.reader_id)

    volume, affine, dicom = load_dicom_volume(series_dir)
    if xml_series_uid and xml_series_uid != dicom["series_instance_uid"]:
        raise ValueError(
            "XML SeriesInstanceUid does not match the supplied CT series: "
            f"XML={xml_series_uid}, DICOM={dicom['series_instance_uid']}"
        )
    if xml_study_uid and dicom["study_instance_uid"] and xml_study_uid != dicom["study_instance_uid"]:
        raise ValueError("XML StudyInstanceUID does not match the supplied DICOM study")
    missing_sops = sorted(set(annotation["sop_instance_uids"]) - set(dicom["sop_to_index"]))
    if missing_sops:
        raise ValueError(
            "Selected annotation references SOP Instance UIDs missing from the supplied CT series: "
            + ", ".join(missing_sops[:5])
        )

    spacing_z = float(dicom["slice_spacing_mm"])
    center_world, used_sops, max_z_delta = annotation_center_world(
        annotation,
        dicom["headers_by_sop"],
        dicom["series_instance_uid"],
        tuple(volume.shape),
        z_tolerance_mm=max(1.0, spacing_z * 0.5 + 0.2),
    )
    center_float_xyz = (np.linalg.inv(affine) @ np.append(center_world, 1.0))[:3]
    center_voxel = np.rint(center_float_xyz).astype(int)
    roi, cube_hu = extract_normalized_roi(volume, center_float_xyz)
    target_world = (affine @ np.append(center_voxel, 1.0))[:3]

    roi_path = output_dir / f"{case_id}_xml_nodule.npy"
    metadata_path = output_dir / f"{case_id}_registration.json"
    preview_path = output_dir / f"{case_id}_registration_preview.png"
    entry_path = output_dir / f"{case_id}_case_entry.json"
    outputs = [roi_path, metadata_path, preview_path, entry_path]
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError("Output already exists; pass --overwrite: " + ", ".join(existing))

    repo_root = Path(__file__).resolve().parent.parent
    case_entry = {
        "case_id": case_id,
        "roi_path": _repo_relative_or_absolute(roi_path, repo_root),
        "ct_series_path": str(series_dir),
        "series_instance_uid": dicom["series_instance_uid"],
        "target_voxel": center_voxel.tolist(),
        "target_mm": target_world.tolist(),
        "annotation_xml_path": str(xml_path),
        "annotation_nodule_id": annotation["nodule_id"],
        "annotation_reader_id": annotation["reader_id"],
    }
    config_update = None
    if args.cases_config is not None:
        config_update = _prepare_config_update(
            args.cases_config.expanduser().resolve(), case_entry, args.replace_case
        )

    metadata = {
        **case_entry,
        "roi_shape_zyx": list(roi.shape),
        "annotation_malignancy_score": annotation["malignancy"],
        "annotation_roi_slice_count": annotation["roi_count"],
        "matched_sop_instance_uids": used_sops,
        "annotation_center_voxel_float_xyz": center_float_xyz.tolist(),
        "annotation_center_patient_lps_mm": center_world.tolist(),
        "target_point_patient_lps_mm": target_world.tolist(),
        "xml_to_dicom_max_z_difference_mm": max_z_delta,
        "ct_volume_shape_zyx": list(volume.shape),
        "roi_intensity_window_hu": [HU_MIN, HU_MAX],
        "roi_normalization": "clip[-1000,400]; (HU + 1000) / 1400",
        "registration_method": "mean per-slice XML edgeMap centroid mapped by SOP UID and DICOM geometry",
        "patient_to_robot_transform": None,
        "patient_to_robot_transform_note": "Not inferred from CT; register a measured transform or explicitly configure a demo placement separately.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(roi_path, roi)
    _write_json(metadata_path, metadata)
    _write_json(entry_path, case_entry)
    _save_preview(preview_path, cube_hu, center_float_xyz)
    if config_update is not None:
        _write_json(args.cases_config.expanduser().resolve(), config_update)
    return {"case_entry": case_entry, "metadata_path": str(metadata_path), "roi_path": str(roi_path), "preview_path": str(preview_path)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-xml", type=Path, required=True, help="One LIDC XML annotation file")
    parser.add_argument("--ct-series", type=Path, help="Exact DICOM series directory for registration")
    parser.add_argument("--case-id", help="Case identifier, e.g. LIDC-IDRI-0041")
    parser.add_argument("--nodule-id", help="Exact noduleID value from the selected XML")
    parser.add_argument("--reader-id", help="servicingRadiologistID when the nodule ID is repeated")
    parser.add_argument("--output-dir", type=Path, default=Path("data/registered_rois"))
    parser.add_argument("--cases-config", type=Path, help="Optionally append/update an existing cases.json")
    parser.add_argument("--replace-case", action="store_true", help="Replace an existing case entry when --cases-config is used")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite ROI/metadata/preview outputs")
    parser.add_argument("--list-annotations", action="store_true", help="List XML nodule IDs and reader IDs, then exit")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        root, _, _ = read_annotation_xml(args.annotation_xml.expanduser().resolve())
        records = annotation_records(root)
        if args.list_annotations:
            for record in records:
                print(
                    f"nodule_id={record['nodule_id']!r}\treader_id={record['reader_id']!r}\t"
                    f"slices={record['roi_count']}\tmalignancy={record['malignancy']}"
                )
            return 0
        missing = [name for name in ("case_id", "nodule_id") if not getattr(args, name)]
        if args.ct_series is None:
            missing.append("ct-series")
        if missing:
            raise ValueError("Required for registration: " + ", ".join("--" + name.replace("_", "-") for name in missing))
        result = register_case(args)
    except Exception as exc:
        print(f"Registration failed: {exc}", file=sys.stderr)
        return 2

    print(f"ROI: {result['roi_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"QC preview: {result['preview_path']}")
    print("Case entry:")
    print(json.dumps(result["case_entry"], indent=2, ensure_ascii=False))
    print("Patient-to-robot calibration was not generated; review the metadata before simulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
