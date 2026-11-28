"""Test prepared TPC data and padding using small synthetic files.

Contributor: Audrey Labrie.
Paper: Rocheteau et al., Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit.
Paper link: https://arxiv.org/abs/2007.09483
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from examples.tpc_los_support import PreparedICUStays, collate_stays


class TestPreparedICUStays(unittest.TestCase):
    """Check loading and batching without eICU data."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="tpc-data-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "train"
        self.folder.mkdir()

        for filename, features in (
            ("timeseries_scaler.json", ["signal_a", "signal_b"]),
            ("flat_encoder.json", ["age"]),
            ("diagnosis_encoder.json", ["diagnosis_a"]),
        ):
            (self.root / filename).write_text(
                json.dumps({"features": features}),
                encoding="utf-8",
            )

        self._write_csv(
            "labels.csv",
            ["patient", "uniquepid", "actualiculos"],
            [
                [202, "synthetic-202", 2.0],
                [101, "synthetic-101", 8.25 / 24],
            ],
        )
        self._write_csv(
            "flat.csv",
            ["patient", "age"],
            [[202, 0.75], [101, 0.25]],
        )
        self._write_csv(
            "diagnoses.csv",
            ["patient", "diagnosis_a"],
            [[101, 1.0], [202, 0.0]],
        )

        self.columns = [
            "patient",
            "time",
            "signal_a",
            "signal_b",
            "signal_a_mask",
            "signal_b_mask",
            "hour",
        ]
        self.rows = [
            [stay, hour, hour, 10 + hour, 1.0, 0.5, 0.25]
            for stay, length in ((101, 8), (202, 6))
            for hour in range(1, length + 1)
        ]
        self._write_csv("timeseries.csv", self.columns, self.rows)

    def _write_csv(
        self,
        filename: str,
        columns: list,
        rows: list,
    ) -> None:
        with (self.folder / filename).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_target_values_and_hour_mask(self) -> None:
        data = PreparedICUStays(self.root, "train")
        self.assertEqual(len(data), 2)

        expected = (
            [
                29 / 96,
                25 / 96,
                21 / 96,
                17 / 96,
                13 / 96,
                9 / 96,
                5 / 96,
                1 / 48,
            ],
            [
                47 / 24,
                46 / 24,
                45 / 24,
                44 / 24,
                43 / 24,
                42 / 24,
            ],
        )
        masks = (
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True],
        )

        for index in range(2):
            with self.subTest(stay=index):
                sample = data[index]

                torch.testing.assert_close(
                    sample["targets"],
                    torch.tensor(expected[index]),
                )
                self.assertEqual(
                    sample["targets"].dtype,
                    torch.float32,
                )
                self.assertEqual(
                    sample["valid_mask"].dtype,
                    torch.bool,
                )
                self.assertEqual(
                    sample["valid_mask"].tolist(),
                    masks[index],
                )

    def test_feature_layout_and_id_alignment(self) -> None:
        data = PreparedICUStays(self.root, "train")

        self.assertEqual(
            (data.features, data.static_dim, data.diagnosis_dim),
            (2, 1, 1),
        )

        for index, stay, length, age, diagnosis in (
            (0, 101, 8, 0.25, 1.0),
            (1, 202, 6, 0.75, 0.0),
        ):
            with self.subTest(stay=stay):
                sample = data[index]

                self.assertEqual(
                    sample["patient_id"],
                    f"synthetic-{stay}",
                )
                self.assertEqual(sample["record_id"], str(stay))
                self.assertEqual(sample["x"].shape, (6, length))
                self.assertEqual(sample["x"].dtype, torch.float32)

                torch.testing.assert_close(
                    sample["x"][:, 0],
                    torch.tensor([
                        1 / 24,
                        1.0,
                        11.0,
                        1.0,
                        0.5,
                        0.25,
                    ]),
                )
                self.assertAlmostEqual(
                    sample["x"][0, -1].item(),
                    length / 24,
                )
                torch.testing.assert_close(
                    sample["static"],
                    torch.tensor([age]),
                )
                torch.testing.assert_close(
                    sample["diagnoses"],
                    torch.tensor([diagnosis]),
                )

    def test_rejects_wrong_column_order(self) -> None:
        order = [0, 1, 3, 2, 4, 5, 6]
        self._write_csv(
            "timeseries.csv",
            [self.columns[i] for i in order],
            [[row[i] for i in order] for row in self.rows],
        )

        with self.assertRaisesRegex(ValueError, "time-series schema"):
            PreparedICUStays(self.root, "train")

    def test_rejects_mismatched_stay_ids(self) -> None:
        self._write_csv(
            "flat.csv",
            ["patient", "age"],
            [[101, 0.25], [999, 0.75]],
        )

        with self.assertRaisesRegex(ValueError, "stay alignment"):
            PreparedICUStays(self.root, "train")

    def test_rejects_interleaved_stays(self) -> None:
        rows = self.rows[:4] + self.rows[8:] + self.rows[4:8]
        self._write_csv("timeseries.csv", self.columns, rows)

        with self.assertRaisesRegex(ValueError, "contiguous by stay"):
            PreparedICUStays(self.root, "train")

    def test_rejects_missing_hour(self) -> None:
        rows = [
            row
            for row in self.rows
            if row[:2] != [101, 3]
        ]
        self._write_csv("timeseries.csv", self.columns, rows)

        with self.assertRaisesRegex(ValueError, "hourly grid"):
            PreparedICUStays(self.root, "train")

    def test_padding_preserves_data_and_excludes_padded_hours(self) -> None:
        data = PreparedICUStays(self.root, "train")
        loader = DataLoader(
            data,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_stays,
            generator=torch.Generator().manual_seed(0),
        )
        batch = next(iter(loader))

        self.assertEqual(batch["x"].shape, (2, 6, 8))
        self.assertEqual(batch["targets"].shape, (2, 8))
        self.assertEqual(batch["valid_mask"].shape, (2, 8))
        self.assertEqual(batch["valid_mask"].dtype, torch.bool)
        self.assertEqual(batch["valid_mask"].sum().item(), 4)

        self.assertEqual(
            batch["valid_mask"].tolist(),
            [
                [False, False, False, False, False, True, True, True],
                [False, False, False, False, False, True, False, False],
            ],
        )

        for index, length in ((0, 8), (1, 6)):
            sample = data[index]

            for key in ("x", "targets", "valid_mask"):
                torch.testing.assert_close(
                    batch[key][index, ..., :length],
                    sample[key],
                )

            for key in ("static", "diagnoses"):
                torch.testing.assert_close(
                    batch[key][index],
                    sample[key],
                )

        torch.testing.assert_close(
            batch["x"][1, :, 6:],
            torch.zeros(6, 2),
        )
        torch.testing.assert_close(
            batch["targets"][1, 6:],
            torch.ones(2),
        )
        self.assertFalse(
            batch["valid_mask"][1, 6:].any().item()
        )


if __name__ == "__main__":
    unittest.main()
