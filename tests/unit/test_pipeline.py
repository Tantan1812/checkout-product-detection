import unittest
from pathlib import Path

from checkout_vision.pipeline import is_image_source, parse_source


class SourceRoutingTests(unittest.TestCase):
    def test_numeric_source_selects_camera_index(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("2"), 2)

    def test_file_source_remains_a_path_string(self):
        self.assertEqual(parse_source("clips/checkout.mp4"), "clips/checkout.mp4")

    def test_image_extensions_are_detected_case_insensitively(self):
        self.assertTrue(is_image_source(Path("sample.JPG")))
        self.assertFalse(is_image_source("checkout.mp4"))
        self.assertFalse(is_image_source(0))


if __name__ == "__main__":
    unittest.main()
