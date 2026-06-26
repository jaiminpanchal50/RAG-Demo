import unittest

from query import build_answer_prompt, detect_language


class QueryHelpersTest(unittest.TestCase):
    def test_detect_language_hindi(self):
        self.assertEqual(detect_language("क्या कीमतें कम हैं?"), "Hindi")

    def test_detect_language_gujarati(self):
        self.assertEqual(detect_language("શું ભાવ સસ્તા છે?"), "Gujarati")

    def test_detect_language_english(self):
        self.assertEqual(detect_language("What is the price?"), "English")

    def test_build_answer_prompt_mentions_target_language(self):
        prompt = build_answer_prompt("શું ભાવ સસ્તા છે?", "Context here", "Gujarati")
        self.assertIn("Gujarati", prompt)
        self.assertIn("Answer", prompt)


if __name__ == "__main__":
    unittest.main()
