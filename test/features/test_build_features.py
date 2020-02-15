import unittest

from numpy.testing import *

from src.features.build_features import *


class add_lag_features_TestCase(unittest.TestCase):
    def test(self):
        # given:
        old_col_name = "my_col"
        lag_col_name = "my_col_2_lag"
        df = pd.DataFrame({
            "site_id": 1,
            "building_id": 2,
            "meter": 3,
            old_col_name: np.arange(4),
        })
        cols = [old_col_name]
        windows = [2]

        # when:
        actual = add_lag_features(df, cols, windows)

        # then:
        assert_array_equal(actual.columns, ["site_id", "building_id", "meter", old_col_name, lag_col_name])
        assert_array_equal(actual[old_col_name], [0,        1,   2,   3])
        assert_array_equal(actual[lag_col_name], [np.nan, 0.5, 1.5, 2.5])


class encode_wind_direction_TestCase(unittest.TestCase):
    def test_nan_degrees(self):
        # given:
        df = pd.DataFrame({
            "wind_direction": [np.nan],
            "wind_speed": [10],
        })

        # when:
        actual = encode_wind_direction(df)

        # then:
        assert_array_equal(actual.columns, ["wind_direction", "wind_speed", "wind_direction_sin", "wind_direction_cos"])
        assert_array_equal(actual["wind_direction_sin"], [0])
        assert_array_equal(actual["wind_direction_cos"], [0])

    def test_0_degrees(self):
        # given:
        df = pd.DataFrame({
            "wind_direction": [0],
            "wind_speed": [10],
        })

        # when:
        actual = encode_wind_direction(df)

        # then:
        assert_array_equal(actual.columns, ["wind_direction", "wind_speed", "wind_direction_sin", "wind_direction_cos"])
        assert_allclose(actual["wind_direction_sin"], [0], atol=1e-15)
        assert_allclose(actual["wind_direction_cos"], [1], atol=1e-15)

    def test_90_degrees(self):
        # given:
        df = pd.DataFrame({
            "wind_direction": [90],
            "wind_speed": [10],
        })

        # when:
        actual = encode_wind_direction(df)

        # then:
        assert_array_equal(actual.columns, ["wind_direction", "wind_speed", "wind_direction_sin", "wind_direction_cos"])
        assert_allclose(actual["wind_direction_sin"], [1], atol=1e-15)
        assert_allclose(actual["wind_direction_cos"], [0], atol=1e-15)

    def test_0_wind_speed(self):
        # given:
        df = pd.DataFrame({
            "wind_direction": [90],
            "wind_speed": [0],
        })

        # when:
        actual = encode_wind_direction(df)

        # then:
        assert_array_equal(actual.columns, ["wind_direction", "wind_speed", "wind_direction_sin", "wind_direction_cos"])
        assert_array_equal(actual["wind_direction_sin"], [0])
        assert_array_equal(actual["wind_direction_cos"], [0])

    def test_nan_wind_speed(self):
        # given:
        df = pd.DataFrame({
            "wind_direction": [90],
            "wind_speed": [np.nan],
        })

        # when:
        actual = encode_wind_direction(df)

        # then:
        assert_array_equal(actual.columns, ["wind_direction", "wind_speed", "wind_direction_sin", "wind_direction_cos"])
        assert_array_equal(actual["wind_direction_sin"], [0])
        assert_array_equal(actual["wind_direction_cos"], [0])
