import argparse
import json
import os
from pathlib import Path
import numpy as np
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
import pydicom
from scipy import ndimage

class VirtualChestEnvironment:
    """
    Virtual chest geometry model that extracts the skin surface and rib
    obstacles (bounding boxes) from a 3D CT scan.
    """
    def __init__(self, ct_path):
        """
        Load a patient CT scan from either a directory containing a DICOM
        series (.dcm) or a supported image file (.mhd / .nii).
        """
        print(f"Loading CT image, please wait: {ct_path}")
        self.series_instance_uid = None
        self._index_to_physical = None
        if os.path.isdir(ct_path):
            # Resolve one explicit series; never silently choose among nested series.
            series_dirs = []
            for root, _, files in os.walk(ct_path):
                if any(name.lower().endswith('.dcm') for name in files):
                    series_dirs.append(root)
            if not series_dirs:
                raise FileNotFoundError(f"No DICOM series found below {ct_path}")
            if len(series_dirs) > 1:
                raise ValueError(
                    "More than one DICOM series is present. Pass the exact series folder, "
                    f"not its parent: {series_dirs}"
                )

            series_dir = series_dirs[0]
            if sitk is not None:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
                reader.SetFileNames(dicom_names)
                self.ct_image = reader.Execute()
                self.spacing = np.array(self.ct_image.GetSpacing())
                self.origin = np.array(self.ct_image.GetOrigin())
                self.direction = np.array(self.ct_image.GetDirection()).reshape(3, 3)
                self.ct_array = sitk.GetArrayFromImage(self.ct_image)
                self.series_instance_uid = pydicom.dcmread(dicom_names[0], stop_before_pixels=True).SeriesInstanceUID
            else:
                self._load_dicom_with_pydicom(series_dir)
        else:
            if sitk is None:
                raise RuntimeError("SimpleITK is required to read a single image file; pass a DICOM series folder instead")
            self.ct_image = sitk.ReadImage(ct_path)
            self.spacing = np.array(self.ct_image.GetSpacing())
            self.origin = np.array(self.ct_image.GetOrigin())
            self.direction = np.array(self.ct_image.GetDirection()).reshape(3, 3)
            self.ct_array = sitk.GetArrayFromImage(self.ct_image)
        print(f"CT loaded. Original array shape [Z, Y, X]: {self.ct_array.shape}")

    def _load_dicom_with_pydicom(self, series_dir):
        files = sorted(Path(series_dir).glob("*.dcm"))
        headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in files]
        if not headers:
            raise FileNotFoundError(f"No DICOM slices found in {series_dir}")

        first = headers[0]
        row_direction = np.asarray(first.ImageOrientationPatient[:3], dtype=float)
        column_direction = np.asarray(first.ImageOrientationPatient[3:], dtype=float)
        normal = np.cross(row_direction, column_direction)
        projections = np.array([
            np.dot(np.asarray(header.ImagePositionPatient, dtype=float), normal)
            for header in headers
        ])
        order = np.argsort(projections)
        files = [files[index] for index in order]
        headers = [headers[index] for index in order]
        projections = projections[order]

        if len(headers) > 1:
            slice_spacing = float(np.median(np.diff(projections)))
        else:
            slice_spacing = float(getattr(first, "SpacingBetweenSlices", first.SliceThickness))
        row_spacing, column_spacing = map(float, first.PixelSpacing)

        orientation = np.column_stack((
            row_direction * column_spacing,
            column_direction * row_spacing,
            normal * slice_spacing,
        ))
        self._index_to_physical = np.eye(4)
        self._index_to_physical[:3, :3] = orientation
        self._index_to_physical[:3, 3] = np.asarray(headers[0].ImagePositionPatient, dtype=float)
        self.origin = self._index_to_physical[:3, 3].copy()
        self.spacing = np.array([column_spacing, row_spacing, slice_spacing])
        self.direction = orientation / self.spacing
        self.series_instance_uid = str(first.SeriesInstanceUID)

        volume = []
        for path, header in zip(files, headers):
            dataset = pydicom.dcmread(path)
            pixels = dataset.pixel_array.astype(np.float32)
            pixels *= float(getattr(dataset, "RescaleSlope", 1.0))
            pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
            volume.append(pixels)
        self.ct_array = np.stack(volume, axis=0)

    def voxel_to_world(self, voxel_coord):
        """
        Convert voxel coordinates (x_idx, y_idx, z_idx) to physical coordinates (X_mm, Y_mm, Z_mm).
        """
        if self._index_to_physical is not None:
            index = np.append(np.asarray(voxel_coord, dtype=float), 1.0)
            return (self._index_to_physical @ index)[:3]
        return np.asarray(
            self.ct_image.TransformIndexToPhysicalPoint([int(c) for c in voxel_coord]),
            dtype=float,
        )

    def world_to_voxel(self, world_coord):
        """
        Convert physical coordinates (X_mm, Y_mm, Z_mm) to voxel indices (x_idx, y_idx, z_idx).
        """
        if self._index_to_physical is not None:
            physical = np.append(np.asarray(world_coord, dtype=float), 1.0)
            return tuple(np.rint(np.linalg.inv(self._index_to_physical) @ physical)[:3].astype(int))
        return self.ct_image.TransformPhysicalPointToIndex(world_coord)

    def extract_skin_surface(self, skin_hu_threshold=-200):
        """
        Extract the chest-wall skin surface by thresholding the body region
        and identifying its outer boundary.
        """
        print(f"Extracting the body-surface mask (HU > {skin_hu_threshold})...")
        # Binarize to separate body tissue from the air background.
        body_mask = (self.ct_array > skin_hu_threshold).astype(np.uint8)
        
        # Fill internal body cavities to obtain a solid body mask.
        # Apply binary hole filling to each 2D slice.
        for z in range(body_mask.shape[0]):
            body_mask[z] = ndimage.binary_fill_holes(body_mask[z])
            
        return body_mask

    def extract_rib_bounding_boxes(self, bone_hu_threshold=300, min_volume_voxels=300):
        """
        Extract 3D rib obstacle bounding boxes by thresholding high-density
        bone and grouping it by connected component. Output axis-aligned
        bounding boxes (AABBs).
        """
        print(f"Extracting the bone mask and computing connected-component boxes (HU > {bone_hu_threshold})...")
        bone_mask = self.ct_array > bone_hu_threshold
        
        # Label 3D connected components.
        labeled_bones, num_features = ndimage.label(bone_mask)
        
        rib_boxes = []
        # Get the slice bounds for each connected component.
        objects = ndimage.find_objects(labeled_bones)
        
        for idx, loc in enumerate(objects):
            if loc is None:
                continue
            
            # Calculate the component volume and filter small noisy bone fragments.
            voxel_count = np.sum(labeled_bones[loc] == (idx + 1))
            if voxel_count < min_volume_voxels:
                continue
            
            z_slice, y_slice, x_slice = loc
            
            # Voxel bounds [min, max].
            # SimpleITK coordinate transforms expect indices in [X, Y, Z] order.
            min_voxel = (x_slice.start, y_slice.start, z_slice.start)
            max_voxel = (x_slice.stop - 1, y_slice.stop - 1, z_slice.stop - 1)
            
            # Convert all eight corners. Using only two corners is incorrect when
            # the image direction matrix contains a rotation or axis flip.
            corners = []
            for x in (min_voxel[0], max_voxel[0]):
                for y in (min_voxel[1], max_voxel[1]):
                    for z in (min_voxel[2], max_voxel[2]):
                        corners.append(self.voxel_to_world((x, y, z)))
            corners = np.asarray(corners, dtype=float)
            min_world = corners.min(axis=0)
            max_world = corners.max(axis=0)
            
            # Axis-aligned 3D bounding box (AABB): [min_x, min_y, min_z, max_x, max_y, max_z].
            box_world = {
                "id": f"rib_cluster_{idx+1}",
                "min_mm": np.minimum(min_world, max_world).tolist(),
                "max_mm": np.maximum(min_world, max_world).tolist(),
                "volume_voxels": int(voxel_count),
                "coordinate_system": "CT physical coordinates in mm",
                "series_instance_uid": self.series_instance_uid,
            }
            rib_boxes.append(box_world)
            
        print(f"✅ Extracted {len(rib_boxes)} major bone/rib obstacle bounding boxes.")
        return rib_boxes

    def _voxel_bounds_to_world(self, min_voxel, max_voxel):
        """Convert inclusive voxel bounds to a conservative physical AABB."""
        corners = []
        for x in (float(min_voxel[0]) - 0.5, float(max_voxel[0]) + 0.5):
            for y in (float(min_voxel[1]) - 0.5, float(max_voxel[1]) + 0.5):
                for z in (float(min_voxel[2]) - 0.5, float(max_voxel[2]) + 0.5):
                    corners.append(self.voxel_to_world((x, y, z)))
        corners = np.asarray(corners, dtype=float)
        return corners.min(axis=0), corners.max(axis=0)

    def extract_local_rib_bounding_boxes(
        self,
        target_voxel,
        bone_hu_threshold=300,
        min_volume_voxels=40,
        x_min_offset_mm=-35.0,
        x_max_offset_mm=140.0,
        y_half_range_mm=150.0,
        z_half_range_mm=80.0,
        segment_slices=5,
        exclude_target_roi_half_size_voxels=0,
        case_id=None,
    ):
        """Extract segmented rib candidates around one ROI from this exact CT series.

        The x offsets intentionally keep the ROI-side lateral chest wall while
        excluding the central spine/sternum for the supplied 0037 orientation.
        This is a simplified simulation obstacle model, not an anatomical rib
        segmentation method.
        """
        target_voxel = np.asarray(target_voxel, dtype=int)
        if target_voxel.shape != (3,):
            raise ValueError("target_voxel must contain [x, y, z]")
        if segment_slices <= 0:
            raise ValueError("segment_slices must be positive")
        if exclude_target_roi_half_size_voxels < 0:
            raise ValueError("exclude_target_roi_half_size_voxels must be non-negative")

        x, y, z = target_voxel.tolist()
        spacing_x, spacing_y, spacing_z = self.spacing
        x0 = max(0, int(np.floor(x + x_min_offset_mm / spacing_x)))
        x1 = min(self.ct_array.shape[2], int(np.ceil(x + x_max_offset_mm / spacing_x)) + 1)
        y0 = max(0, int(np.floor(y - y_half_range_mm / spacing_y)))
        y1 = min(self.ct_array.shape[1], int(np.ceil(y + y_half_range_mm / spacing_y)) + 1)
        z0 = max(0, int(np.floor(z - z_half_range_mm / spacing_z)))
        z1 = min(self.ct_array.shape[0], int(np.ceil(z + z_half_range_mm / spacing_z)) + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            raise ValueError("The requested local CT crop is empty")

        local_mask = self.ct_array[z0:z1, y0:y1, x0:x1] > bone_hu_threshold
        if exclude_target_roi_half_size_voxels:
            half = int(exclude_target_roi_half_size_voxels)
            target_x0 = max(x0, x - half)
            target_x1 = min(x1, x + half)
            target_y0 = max(y0, y - half)
            target_y1 = min(y1, y + half)
            target_z0 = max(z0, z - half)
            target_z1 = min(z1, z + half)
            local_mask[
                target_z0 - z0 : target_z1 - z0,
                target_y0 - y0 : target_y1 - y0,
                target_x0 - x0 : target_x1 - x0,
            ] = False
        structure = ndimage.generate_binary_structure(3, 3)
        labeled, num_features = ndimage.label(local_mask, structure=structure)
        objects = ndimage.find_objects(labeled)
        target_world = self.voxel_to_world(target_voxel)
        boxes = []
        next_id = 1

        for label_id, loc in enumerate(objects, start=1):
            if loc is None:
                continue
            component_count = int(np.count_nonzero(labeled[loc] == label_id))
            if component_count < min_volume_voxels:
                continue

            local_z, local_y, local_x = np.where(labeled == label_id)
            global_z = local_z + z0
            global_y = local_y + y0
            global_x = local_x + x0
            component_z_min = int(global_z.min())
            component_z_max = int(global_z.max())

            for segment_start in range(component_z_min, component_z_max + 1, segment_slices):
                segment_stop = min(segment_start + segment_slices - 1, component_z_max)
                selected = (global_z >= segment_start) & (global_z <= segment_stop)
                if int(np.count_nonzero(selected)) < min_volume_voxels:
                    continue

                min_voxel = (
                    int(global_x[selected].min()),
                    int(global_y[selected].min()),
                    int(global_z[selected].min()),
                )
                max_voxel = (
                    int(global_x[selected].max()),
                    int(global_y[selected].max()),
                    int(global_z[selected].max()),
                )
                min_world, max_world = self._voxel_bounds_to_world(min_voxel, max_voxel)
                boxes.append(
                    {
                        "id": f"rib_candidate_{next_id:03d}",
                        "min_mm": min_world.tolist(),
                        "max_mm": max_world.tolist(),
                        "min_voxel": list(min_voxel),
                        "max_voxel": list(max_voxel),
                        "volume_voxels": int(np.count_nonzero(selected)),
                        "coordinate_system": "CT physical coordinates in mm",
                        "source_case_id": case_id or "unspecified",
                        "series_instance_uid": self.series_instance_uid,
                        "target_voxel": target_voxel.tolist(),
                        "target_point_mm": target_world.tolist(),
                        "method": "local_HU_threshold_3D_components_segmented_AABB",
                        "bone_hu_threshold": float(bone_hu_threshold),
                        "excluded_target_roi_half_size_voxels": int(exclude_target_roi_half_size_voxels),
                        "crop_voxel_bounds": {
                            "min": [x0, y0, z0],
                            "max": [x1 - 1, y1 - 1, z1 - 1],
                        },
                    }
                )
                next_id += 1

        boxes.sort(key=lambda item: (item["min_voxel"][2], item["min_voxel"][1], item["min_voxel"][0]))
        print(
            f"✅ Local rib candidate extraction complete: {len(boxes)} segmented obstacle boxes; "
            f"crop Z/Y/X=[{z0}:{z1}, {y0}:{y1}, {x0}:{x1}], "
            f"components={num_features}"
        )
        return boxes

    def save_local_extraction_preview(self, output_path, target_voxel, boxes, z_indices=None):
        """Save axial CT slices with HU mask, ROI and extracted AABBs overlaid."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        target_voxel = np.asarray(target_voxel, dtype=int)
        if z_indices is None:
            z_indices = [
                max(0, int(target_voxel[2]) - 14),
                int(target_voxel[2]),
                min(self.ct_array.shape[0] - 1, int(target_voxel[2]) + 14),
            ]

        fig, axes = plt.subplots(1, len(z_indices), figsize=(6 * len(z_indices), 6), squeeze=False)
        for axis, slice_index in zip(axes[0], z_indices):
            image = np.clip(self.ct_array[int(slice_index)], -1000, 400)
            axis.imshow(image, cmap="gray", vmin=-1000, vmax=400)
            bone = self.ct_array[int(slice_index)] > 300
            axis.imshow(np.ma.masked_where(~bone, bone), cmap="autumn", alpha=0.20)
            axis.plot(target_voxel[0], target_voxel[1], "c+", markersize=12, markeredgewidth=2)
            for box in boxes:
                min_v = box["min_voxel"]
                max_v = box["max_voxel"]
                if min_v[2] <= int(slice_index) <= max_v[2]:
                    axis.add_patch(
                        Rectangle(
                            (min_v[0], min_v[1]),
                            max_v[0] - min_v[0] + 1,
                            max_v[1] - min_v[1] + 1,
                            fill=False,
                            edgecolor="lime",
                            linewidth=1.2,
                        )
                    )
            axis.set_title(f"z index {slice_index}, physical z {self.voxel_to_world((0, 0, slice_index))[2]:.1f} mm")
            axis.set_xlim(0, self.ct_array.shape[2] - 1)
            axis.set_ylim(self.ct_array.shape[1] - 1, 0)
            axis.set_xlabel("x voxel")
            axis.set_ylabel("y voxel")
        fig.suptitle("Local CT-derived rib candidates: cyan=ROI, lime=segmented AABB")
        fig.tight_layout()
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Saved local extraction preview to: {output_path.resolve()}")

# ==========================================
# Local testing and JSON export.
# ==========================================
def _parse_args():
    parser = argparse.ArgumentParser(description="Extract CT-space bone AABBs from one DICOM series.")
    parser.add_argument("--ct-path", required=True, help="Folder containing one DICOM series, or a supported image file")
    parser.add_argument(
        "--output",
        default="data/demo_nodules/rib_obstacles_0037.json",
        help="Output JSON path; coordinates are CT physical coordinates in mm",
    )
    parser.add_argument("--bone-hu-threshold", type=float, default=None)
    parser.add_argument("--min-volume-voxels", type=int, default=None)
    parser.add_argument(
        "--target-voxel",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        help="ROI center voxel in the selected CT series; enables local segmented extraction",
    )
    parser.add_argument("--series-instance-uid", help="Refuse to run if the selected DICOM series UID differs")
    parser.add_argument("--preview-output", help="PNG path for local extraction quality-control preview")
    parser.add_argument("--segment-slices", type=int, default=None)
    parser.add_argument("--x-min-offset-mm", type=float, default=-35.0)
    parser.add_argument("--x-max-offset-mm", type=float, default=140.0)
    parser.add_argument("--y-half-range-mm", type=float, default=150.0)
    parser.add_argument("--z-half-range-mm", type=float, default=80.0)
    parser.add_argument("--case-id", default="LIDC-IDRI-0037")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ct_path = Path(args.ct_path).expanduser()
    if not ct_path.exists():
        raise FileNotFoundError(f"CT path does not exist: {ct_path}")

    env = VirtualChestEnvironment(str(ct_path))
    if args.series_instance_uid and env.series_instance_uid != args.series_instance_uid:
        raise ValueError(
            "SeriesInstanceUID mismatch: "
            f"selected={env.series_instance_uid}, expected={args.series_instance_uid}"
        )
    if args.target_voxel is not None:
        # Axial slice boxes are deliberately conservative but avoid the large
        # empty-space envelopes produced when curved ribs are merged across z.
        bone_hu_threshold = 500.0 if args.bone_hu_threshold is None else args.bone_hu_threshold
        min_volume_voxels = 40 if args.min_volume_voxels is None else args.min_volume_voxels
        segment_slices = 5 if args.segment_slices is None else args.segment_slices
        rib_boxes = env.extract_local_rib_bounding_boxes(
            target_voxel=args.target_voxel,
            bone_hu_threshold=bone_hu_threshold,
            min_volume_voxels=min_volume_voxels,
            x_min_offset_mm=args.x_min_offset_mm,
            x_max_offset_mm=args.x_max_offset_mm,
            y_half_range_mm=args.y_half_range_mm,
            z_half_range_mm=args.z_half_range_mm,
            segment_slices=segment_slices,
            case_id=args.case_id,
        )
        if args.preview_output:
            env.save_local_extraction_preview(args.preview_output, args.target_voxel, rib_boxes)
    else:
        rib_boxes = env.extract_rib_bounding_boxes(
            bone_hu_threshold=300.0 if args.bone_hu_threshold is None else args.bone_hu_threshold,
            min_volume_voxels=300 if args.min_volume_voxels is None else args.min_volume_voxels,
        )
    if not rib_boxes:
        raise RuntimeError("No bone components passed the configured threshold; refusing to write an empty obstacle file")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rib_boxes, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(rib_boxes)} CT-space obstacles to: {output.resolve()}")
