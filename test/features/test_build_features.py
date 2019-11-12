import unittest

import pandas as pd

from src.features import build_features


class encode_categorical_data_TestCase(unittest.TestCase):
    def test(self):
        # given:
        meter = [0, 1]
        primary_use = ["A", "B"]
        df = pd.DataFrame(list(zip(meter, primary_use)), columns=['meter', 'primary_use'])

        # when:
        df = build_features.encode_categorical_data(df)

        # then:
        self.assertEqual(list(df.columns), ["meter_0", "meter_1", "primary_use_A", "primary_use_B"])
        self.assertEqual(df.values.tolist(), [[1, 0, 1, 0], [0, 1, 0, 1]])
