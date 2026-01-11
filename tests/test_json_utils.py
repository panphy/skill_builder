import unittest

from utils.json_utils import safe_parse_json


class JsonUtilsTests(unittest.TestCase):
    """Usage note: update safe_parse_json only in utils/json_utils.py."""

    def test_safe_parse_json_prefers_steps_in_fenced_block(self) -> None:
        text = """
        Here is the output:
        ```json
        {"foo": 1}
        ```
        And later:
        ```json
        {"steps": ["a", "b"], "bar": 2}
        ```
        """
        parsed = safe_parse_json(text)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("steps"), ["a", "b"])

    def test_safe_parse_json_finds_balanced_object(self) -> None:
        text = "prefix {\"alpha\": 3, \"beta\": 4} suffix"
        parsed = safe_parse_json(text)
        self.assertEqual(parsed, {"alpha": 3, "beta": 4})


if __name__ == "__main__":
    unittest.main()
