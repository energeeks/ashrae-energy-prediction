import os
import shutil
import tempfile
import unittest

from src.data.make_dataset import main as make_dataset
from src.features.build_features import main as build_features
from src.models.train_model import main as train_model
from src.models.predict_model import main as predict_model


class main_TestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_lgbm_cv(self):
        self.parameterized_test("lgbm", "cv")

    def test_lgbm_full(self):
        self.parameterized_test("lgbm", "full")

    def parameterized_test(self, model, mode):
        # given:
        data_dir = "test-data"
        interim_dir = self.test_dir + "/interim"
        processed_dir = self.test_dir + "/processed"
        model_dir = self.test_dir + "/model"
        model_path = model_dir + ("" if mode == "full" else "_" + mode) + "/0001.txt"
        submission_dir = self.test_dir + "/submissions"
        submission_path = submission_dir + "/submission.csv"

        # data preparation
        # when:
        make_dataset(data_dir, interim_dir)

        # then:
        self.assertTrue(os.path.exists(interim_dir + "/test_data.pkl"))
        self.assertTrue(os.path.exists(interim_dir + "/test_data.pkl"))

        # feature engineering
        # when:
        build_features(data_dir, processed_dir)

        # then:
        self.assertTrue(os.path.exists(processed_dir + "/test_data.pkl"))
        self.assertTrue(os.path.exists(processed_dir + "/test_data.pkl"))

        # model training
        # when:
        train_model(model, mode, processed_dir, model_dir)

        # then:
        self.assertTrue(os.path.exists(model_path))

        # model prediction
        # when:
        predict_model(processed_dir, model, model_path, submission_path)

        # then:
        self.assertTrue(os.path.exists(submission_path))
