import unittest
# from src.extend_dataset import extend_set
from src.aumentation_dataset import process_categories
from src.aumentation_dataset import process_images


class ExtendDatasetTest(unittest.TestCase):

    def test_one(self):
        # extend_set()
        process_categories()
        # process_images()




