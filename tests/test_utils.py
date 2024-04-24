from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from comsol.cmdline import main
from comsol.utils import BandDataset


class FieldDatasetTest(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        self.params = [
            {"param1": "value1", "param2": "value2", "param3": "value3"},
            {"param1": "value4", "param2": "value5", "param3": "value6"},
            # Add more sample parameters if needed
        ]
        self.res = np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 2, 4, 6, 8, 10],
                # Add more sample results if needed
            ]
        )
        self.Bs_test = [[x, y] for x in (0, 1) for y in range(1, 7)]
        self.dataset = BandDataset(saved_path="exports/saved")

    def test_get_Bs(self):
        # Test the get_Bs method
        expected_result = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
        result = self.dataset.get_Bs(np.array(self.Bs_test))
        np.testing.assert_array_equal(result, expected_result)

    def test_to_rand(self):
        # Test the to_rand method
        params = {"param1": "0.5", "param2": "1.0", "param3": "1.5"}
        expected_result = np.array([0.5 / 360, 1.0, 1.5])
        result = self.dataset.to_rand(params)
        np.testing.assert_array_equal(result, expected_result)

    def test_normalization(self):
        # Test the normalization method
        arr = np.array([1, 2, 3, 4, 5])
        expected_result = (arr - 1) / 4, 1, 5
        result = self.dataset.normalization(arr)
        np.testing.assert_array_equal(result[0], expected_result[0])
        self.assertEqual(result[1], expected_result[1])
        self.assertEqual(result[2], expected_result[2])

    def test_len(self):
        # Test the __len__ method
        expected_result = len(list(Path("exports/saved").glob("*.pkl")))
        result = len(self.dataset)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
