import unittest
from core.data_access.lakehouse_connector import LakehouseConnector
from core.security.sql_validator import SQLValidator

class TestLakehouseSecurity(unittest.TestCase):
    def setUp(self):
        self.connector = LakehouseConnector()

    def test_block_drop(self):
        result = self.connector.execute("DROP TABLE financials")
        self.assertIn("SecurityViolation", result)

    def test_block_delete(self):
        result = self.connector.execute("DELETE FROM financials")
        self.assertIn("SecurityViolation", result)

    def test_block_update(self):
        result = self.connector.execute("UPDATE financials SET value=0")
        self.assertIn("SecurityViolation", result)

    def test_block_insert(self):
        result = self.connector.execute("INSERT INTO financials VALUES (1, 2, 3)")
        self.assertIn("SecurityViolation", result)

    def test_block_alter(self):
        result = self.connector.execute("ALTER TABLE financials ADD COLUMN hacked int")
        self.assertIn("SecurityViolation", result)

    def test_block_truncate(self):
        result = self.connector.execute("TRUNCATE TABLE financials")
        self.assertIn("SecurityViolation", result)

    def test_block_grant(self):
        result = self.connector.execute("GRANT ALL ON financials TO public")
        self.assertIn("SecurityViolation", result)

    def test_block_semicolon_chaining(self):
        result = self.connector.execute("SELECT * FROM financials; DROP TABLE financials")
        self.assertIn("SecurityViolation", result)

    def test_allow_valid_select(self):
        result = self.connector.execute("SELECT * FROM financials")
        self.assertNotIn("SecurityViolation", result)

    def test_allow_select_with_newlines(self):
        result = self.connector.execute("""
            SELECT *
            FROM financials
        """)
        self.assertNotIn("SecurityViolation", result)

    def test_allow_semicolon_in_string(self):
        # This is the key test that regex often fails
        result = self.connector.execute("SELECT * FROM financials WHERE name = 'semi;colon'")
        self.assertNotIn("SecurityViolation", result)

    def test_allow_select_with_comments(self):
        # Comments should be stripped/ignored and not block valid queries
        # But if the comment HIDES a bad command, it should still be blocked (conceptually).
        # Our validator verifies the first meaningful token is SELECT.
        result = self.connector.execute("""
            -- This is a comment
            SELECT * FROM financials
        """)
        self.assertNotIn("SecurityViolation", result)

    def test_block_comment_hiding_command(self):
        # If we just stripped comments and then regexed, this might pass if not careful.
        # But here we tokenize.
        # "/* comment */ DROP TABLE" -> tokens: [DROP, TABLE] -> Fail
        result = self.connector.execute("/* harmless comment */ DROP TABLE financials")
        self.assertIn("SecurityViolation", result)

    def test_block_cte(self):
        # We currently block CTEs (WITH ...) as per strict whitelist rules.
        result = self.connector.execute("WITH cte AS (SELECT * FROM t) SELECT * FROM cte")
        self.assertIn("SecurityViolation", result)

if __name__ == '__main__':
    unittest.main()
