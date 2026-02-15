import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
import os

# Add shared library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from wikipedia.client import download_pageviews, download_pageviews_range

class TestClient(unittest.TestCase):
    @patch('requests.get')
    def test_download_pageviews(self, mock_get):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [{
                "project": "en.wikipedia",
                "access": "all-access",
                "year": "2025",
                "month": "01",
                "day": "15",
                "articles": [
                    {"article": "Main_Page", "views": 100, "rank": 1},
                    {"article": "Python_(programming_language)", "views": 50, "rank": 2}
                ]
            }]
        }
        mock_get.return_value = mock_response

        date = datetime(2025, 1, 15)
        result = download_pageviews(date)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['article'], "Main_Page")
        self.assertEqual(result[0]['date'], "2025-01-15")
        self.assertEqual(result[1]['article'], "Python_(programming_language)")

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn("2025/01/15", args[0])

    @patch('requests.get')
    def test_download_pageviews_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        date = datetime(2025, 1, 15)
        with self.assertRaises(Exception) as cm:
            download_pageviews(date)

        self.assertIn("404", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
