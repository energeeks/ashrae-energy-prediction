import unittest

from src.models.predict_model import *


class get_submission_error_TestCase(unittest.TestCase):
    def test_valid_submission(self):
        # given:
        df = pd.DataFrame({
            "row_id": np.arange(41697600),
            "meter_reading": np.zeros(41697600),
        })

        # when:
        error = get_submission_error(df)

        # then:
        self.assertIsNone(error)

    def test_incorrect_column_names(self):
        # given:
        df = pd.DataFrame({
            "row": np.arange(41697600),
            "meter_reading": np.zeros(41697600),
        })

        # when:
        actual = get_submission_error(df)

        # then:
        expected = "Submission has incorrect columns: ['row', 'meter_reading'], expected: ['row_id', 'meter_reading']"
        self.assertEqual(expected, actual)

    def test_incorrect_column_order(self):
        # given:
        df = pd.DataFrame({
            "meter_reading": np.zeros(41697600),
            "row_id": np.arange(41697600),
        })

        # when:
        actual = get_submission_error(df)

        # then:
        expected = "Submission has incorrect columns: ['meter_reading', 'row_id'], expected: ['row_id', 'meter_reading']"
        self.assertEqual(expected, actual)

    def test_to_few_row_ids(self):
        # given:
        df = pd.DataFrame({
            "row_id": np.arange(41697599),
            "meter_reading": np.zeros(41697599),
        })

        # when:
        actual = get_submission_error(df)

        # then:
        expected = "Submission has to few rows: 41697599, expected: 41697600"
        self.assertEqual(expected, actual)

    def test_to_many_row_ids(self):
        # given:
        df = pd.DataFrame({
            "row_id": np.arange(41697601),
            "meter_reading": np.zeros(41697601),
        })

        # when:
        actual = get_submission_error(df)

        # then:
        expected = "Submission has to many rows: 41697601, expected: 41697600"
        self.assertEqual(expected, actual)

    def test_incorrect_row_ids(self):
        # given:
        df = pd.DataFrame({
            "row_id": np.arange(41697600) * 2,
            "meter_reading": np.zeros(41697600),
        })

        # when:
        actual = get_submission_error(df)

        # then:
        expected = "Submission has incorrect row_ids"
        self.assertEqual(expected, actual)
