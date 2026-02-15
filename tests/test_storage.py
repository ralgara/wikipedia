import unittest
from datetime import datetime
import sys
import os

# Add shared library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from wikipedia.storage import generate_storage_key, parse_storage_key

class TestStorage(unittest.TestCase):
    def test_generate_storage_key(self):
        date = datetime(2025, 1, 15)
        expected = "wikipedia/pageviews/year=2025/month=01/day=15/pageviews_20250115.json"
        self.assertEqual(generate_storage_key(date), expected)

        custom_prefix = "custom/path"
        expected_custom = "custom/path/year=2025/month=01/day=15/pageviews_20250115.json"
        self.assertEqual(generate_storage_key(date, prefix=custom_prefix), expected_custom)

    def test_parse_storage_key(self):
        key = "wikipedia/pageviews/year=2025/month=01/day=15/pageviews_20250115.json"
        expected = datetime(2025, 1, 15)
        self.assertEqual(parse_storage_key(key), expected)

        invalid_key = "invalid/path/data.json"
        self.assertIsNone(parse_storage_key(invalid_key))

if __name__ == '__main__':
    unittest.main()
