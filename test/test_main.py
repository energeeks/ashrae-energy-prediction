import os
import unittest
import tempfile
import shutil

from src.data.make_dataset import main as make_dataset
from src.features.build_features import main as build_features


class main_TestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test(self):
        # given:
        os.chdir("..")
        data_dir = "test-data"
        interim_dir = self.test_dir + "/interim"
        processed_dir = self.test_dir + "/processed"

        # when:
        make_dataset(data_dir, interim_dir)

        # then:
        self.assertTrue(os.path.exists(interim_dir + "/test_data.pkl"))
        self.assertTrue(os.path.exists(interim_dir + "/test_data.pkl"))

        # when:
        build_features(data_dir, processed_dir)

        # then:
        self.assertTrue(os.path.exists(processed_dir + "/test_data.pkl"))
        self.assertTrue(os.path.exists(processed_dir + "/test_data.pkl"))
