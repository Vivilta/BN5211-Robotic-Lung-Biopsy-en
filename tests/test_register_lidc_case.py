import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from register_lidc_case import (
    annotation_center_world,
    annotation_records,
    extract_normalized_roi,
    _prepare_config_update,
    select_annotation,
)


def _annotation_xml(duplicate=False):
    second = """
    <readingSession><servicingRadiologistID>reader-B</servicingRadiologistID>
      <unblindedReadNodule><noduleID>Nodule 1</noduleID><roi>
        <imageSOP_UID>sop-1</imageSOP_UID><inclusion>TRUE</inclusion>
        <imageZposition>-5</imageZposition><edgeMap><xCoord>2</xCoord><yCoord>4</yCoord></edgeMap>
      </roi></unblindedReadNodule>
    </readingSession>
    """ if duplicate else ""
    return ET.fromstring(f"""<LidcReadMessage>
      <readingSession><servicingRadiologistID>reader-A</servicingRadiologistID>
        <unblindedReadNodule><noduleID>Nodule 1</noduleID><characteristics><malignancy>4</malignancy></characteristics>
          <roi><imageSOP_UID>sop-1</imageSOP_UID><inclusion>TRUE</inclusion><imageZposition>-5</imageZposition>
            <edgeMap><xCoord>2</xCoord><yCoord>4</yCoord></edgeMap>
            <edgeMap><xCoord>4</xCoord><yCoord>8</yCoord></edgeMap>
          </roi>
        </unblindedReadNodule>
      </readingSession>{second}</LidcReadMessage>""")


class RegisterLidcCaseTests(unittest.TestCase):
    def test_annotation_is_selected_by_nodule_and_reader(self):
        records = annotation_records(_annotation_xml())
        selected = select_annotation(records, "Nodule 1")
        self.assertEqual(selected["reader_id"], "reader-A")
        self.assertEqual(selected["malignancy"], 4)
        self.assertEqual(selected["roi_count"], 1)

    def test_duplicate_nodule_ids_require_reader_id(self):
        records = annotation_records(_annotation_xml(duplicate=True))
        with self.assertRaisesRegex(ValueError, "Specify --reader-id"):
            select_annotation(records, "Nodule 1")
        selected = select_annotation(records, "Nodule 1", "reader-B")
        self.assertEqual(selected["reader_id"], "reader-B")

    def test_xml_contour_centroid_maps_through_sop_dicom_geometry(self):
        annotation = select_annotation(annotation_records(_annotation_xml()), "Nodule 1")
        header = SimpleNamespace(
            SeriesInstanceUID="series-1",
            Rows=16,
            Columns=16,
            ImagePositionPatient=[1.0, 2.0, -5.0],
            ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            PixelSpacing=[0.5, 0.8],
        )
        center, sops, z_delta = annotation_center_world(
            annotation,
            {"sop-1": header},
            "series-1",
            (8, 16, 16),
            z_tolerance_mm=1.0,
        )
        np.testing.assert_allclose(center, [3.4, 5.0, -5.0])
        self.assertEqual(sops, ["sop-1"])
        self.assertEqual(z_delta, 0.0)

    def test_unmatched_sop_and_wrong_xml_z_are_rejected(self):
        annotation = select_annotation(annotation_records(_annotation_xml()), "Nodule 1")
        with self.assertRaisesRegex(ValueError, "not present"):
            annotation_center_world(annotation, {}, "series-1", (8, 16, 16), 1.0)
        header = SimpleNamespace(
            SeriesInstanceUID="series-1",
            Rows=16,
            Columns=16,
            ImagePositionPatient=[1.0, 2.0, -10.0],
            ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            PixelSpacing=[0.5, 0.8],
        )
        with self.assertRaisesRegex(ValueError, "imageZposition differs"):
            annotation_center_world(annotation, {"sop-1": header}, "series-1", (8, 16, 16), 1.0)

    def test_roi_is_centered_normalized_and_refuses_boundary_padding(self):
        volume = np.full((40, 40, 40), -1200.0, dtype=np.float32)
        volume[4:36, 4:36, 4:36] = 1000.0
        roi, crop_hu = extract_normalized_roi(volume, [20.0, 20.0, 20.0])
        self.assertEqual(roi.shape, (32, 32, 32))
        self.assertEqual(float(roi.min()), 1.0)
        self.assertEqual(float(roi.max()), 1.0)
        self.assertEqual(float(crop_hu.min()), 1000.0)
        with self.assertRaisesRegex(ValueError, "exceeds CT bounds"):
            extract_normalized_roi(volume, [8.0, 20.0, 20.0])

    def test_config_update_requires_explicit_replacement(self):
        case = {"case_id": "LIDC-IDRI-0001", "roi_path": "new.npy"}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "cases.json"
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {},
                        "cases": [
                            {
                                "case_id": "LIDC-IDRI-0001",
                                "roi_path": "old.npy",
                                "patient_to_robot_base": [[1, 0, 0, 0]] * 4,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "already in"):
                _prepare_config_update(config_path, case, replace_case=False)
            updated = _prepare_config_update(config_path, case, replace_case=True)
        self.assertEqual(len(updated["cases"]), 1)
        self.assertEqual(updated["cases"][0]["roi_path"], "new.npy")
        self.assertIn("patient_to_robot_base", updated["cases"][0])


if __name__ == "__main__":
    unittest.main()
