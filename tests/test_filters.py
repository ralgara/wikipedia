import unittest
import sys
import os

# Add shared library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from wikipedia.filters import is_content, should_flag_for_review, get_hide_reason

class TestFilters(unittest.TestCase):
    def test_is_content(self):
        self.assertTrue(is_content("Python_(programming_language)"))
        self.assertTrue(is_content("Artificial_intelligence"))
        self.assertFalse(is_content("Main_Page"))
        self.assertFalse(is_content("Special:Search"))
        self.assertFalse(is_content("Category:Artificial_intelligence"))
        self.assertFalse(is_content("User:Example"))
        self.assertFalse(is_content("Wikipedia:About"))
        self.assertFalse(is_content("Talk:Python_(programming_language)"))
        self.assertFalse(is_content("Python_talk:Example"))

    def test_should_flag_for_review(self):
        self.assertTrue(should_flag_for_review("Pornography"))
        self.assertTrue(should_flag_for_review("Sexual_intercourse"))
        self.assertTrue(should_flag_for_review("Hentai"))
        self.assertFalse(should_flag_for_review("Python_(programming_language)"))
        self.assertFalse(should_flag_for_review("Cat"))

    def test_get_hide_reason(self):
        self.assertEqual(get_hide_reason("Main_Page"), "main_page")
        self.assertEqual(get_hide_reason("Special:Search"), "special_page")
        self.assertEqual(get_hide_reason("Talk:Python"), "talk_page")
        self.assertEqual(get_hide_reason("Category:Science"), "non_content_page")
        self.assertEqual(get_hide_reason("Pornography"), "flagged_for_review")
        self.assertIsNone(get_hide_reason("Python_(programming_language)"))

if __name__ == '__main__':
    unittest.main()
