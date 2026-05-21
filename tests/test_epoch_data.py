from __future__ import annotations

import io
import tempfile
import unittest
import zipfile

import pandas as pd

from epoch_data import filter_epoch_gpu_clusters, load_epoch_gpu_clusters_from_zip


class EpochDataTests(unittest.TestCase):
    def test_load_epoch_gpu_clusters_from_zip_reads_first_csv(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as archive:
            archive.writestr(
                "gpu_clusters.csv",
                "Country,Total number of AI chips\nFrance,10\n",
            )
        df = load_epoch_gpu_clusters_from_zip(buffer.getvalue())
        self.assertEqual(df.to_dict(orient="records"), [{"Country": "France", "Total number of AI chips": 10}])

    def test_filter_epoch_gpu_clusters_removes_us_china_empty_and_null_rows(self) -> None:
        df = pd.DataFrame(
            [
                {"Country": "France", "Total number of AI chips": 10},
                {"Country": "United States of America", "Total number of AI chips": 20},
                {"Country": "China", "Total number of AI chips": 30},
                {"Country": "Germany", "Total number of AI chips": ""},
                {"Country": "Japan", "Total number of AI chips": None},
            ]
        )

        filtered = filter_epoch_gpu_clusters(df)
        self.assertEqual(filtered.to_dict(orient="records"), [{"Country": "France", "Total number of AI chips": 10}])


if __name__ == "__main__":
    unittest.main()
