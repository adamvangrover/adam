import unittest
from core.security.shield import InputShield

class TestInputShield(unittest.TestCase):

    def test_validate_ticker(self):
        self.assertTrue(InputShield.validate_ticker("AAPL"))
        self.assertTrue(InputShield.validate_ticker("BRK.B"))
        self.assertTrue(InputShield.validate_ticker("BTC-USD"))

        self.assertFalse(InputShield.validate_ticker("INVALID_TICKER_TOO_LONG"))
        self.assertFalse(InputShield.validate_ticker("DROP TABLE"))
        self.assertFalse(InputShield.validate_ticker("<script>"))
        self.assertFalse(InputShield.validate_ticker(""))
        self.assertFalse(InputShield.validate_ticker(None))

    def test_validate_filename(self):
        self.assertTrue(InputShield.validate_filename("report.json"))
        self.assertTrue(InputShield.validate_filename("my_data_2026.csv"))

        self.assertFalse(InputShield.validate_filename("../secret.env"))
        self.assertFalse(InputShield.validate_filename("/etc/passwd"))
        self.assertFalse(InputShield.validate_filename("report.exe"))
        self.assertFalse(InputShield.validate_filename("image.png")) # Only whitelisted extensions

    def test_validate_username(self):
        self.assertTrue(InputShield.validate_username("valid_user"))
        self.assertTrue(InputShield.validate_username("user-123"))

        self.assertFalse(InputShield.validate_username("usr")) # Too short
        self.assertFalse(InputShield.validate_username("user@name")) # Invalid char
        self.assertFalse(InputShield.validate_username("admin<script>"))

    def test_sanitize_text(self):
        raw = "<script>alert('xss')</script>"
        clean = InputShield.sanitize_text(raw)
        self.assertEqual(clean, "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;")

        long_text = "a" * 6000
        truncated = InputShield.sanitize_text(long_text, max_length=10)
        self.assertEqual(len(truncated), 10)

if __name__ == '__main__':
    unittest.main()
