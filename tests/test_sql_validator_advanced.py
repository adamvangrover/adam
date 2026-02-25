import unittest
from core.security.sql_validator import SQLValidator

class TestSQLValidatorAdvanced(unittest.TestCase):
    """
    Advanced security tests for SQLValidator.
    """

    def test_block_xp_cmdshell(self):
        """Should block dangerous stored procedures like xp_cmdshell."""
        query = "SELECT * FROM master..xp_cmdshell('calc.exe')"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block xp_cmdshell")

    def test_block_exec(self):
        """Should block EXEC usage."""
        query = "EXEC('DROP TABLE users')"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block EXEC")

    def test_block_execute(self):
        """Should block EXECUTE usage."""
        query = "EXECUTE('DROP TABLE users')"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block EXECUTE")

    def test_block_into_clause(self):
        """Should block SELECT ... INTO ... (table creation)."""
        query = "SELECT * INTO new_table FROM old_table"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block SELECT INTO")

    def test_block_waitfor(self):
        """Should block WAITFOR DELAY (time-based injection)."""
        query = "SELECT * FROM users; WAITFOR DELAY '0:0:5'"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block WAITFOR")

    def test_block_show(self):
        """Should block SHOW commands."""
        query = "SHOW TABLES"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block SHOW")

    def test_block_describe(self):
        """Should block DESCRIBE commands."""
        query = "DESCRIBE users"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block DESCRIBE")

    def test_block_merge(self):
        """Should block MERGE statements."""
        query = "MERGE INTO target USING source ON (target.id = source.id) WHEN MATCHED THEN UPDATE SET target.val = source.val"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block MERGE")

    def test_allow_complex_select(self):
        """Should allow complex but valid SELECT statements."""
        query = """
        SELECT u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.total > 1000
        GROUP BY u.name
        HAVING count(*) > 5
        ORDER BY o.total DESC
        """
        self.assertTrue(SQLValidator.validate_read_only(query), "Failed to allow valid complex SELECT")

    def test_allow_select_strings_containing_keywords(self):
        """Should allow dangerous keywords inside string literals."""
        query = "SELECT 'DROP TABLE users', 'EXEC xp_cmdshell' FROM logs"
        self.assertTrue(SQLValidator.validate_read_only(query), "Failed to allow keywords in strings")

    def test_allow_select_comments_containing_keywords(self):
        """Should allow dangerous keywords inside comments."""
        query = "SELECT * FROM users -- This query does not EXEC anything"
        self.assertTrue(SQLValidator.validate_read_only(query), "Failed to allow keywords in comments")

    def test_block_case_insensitive(self):
        """Should block dangerous keywords regardless of case."""
        query = "select * from users; Drop Table users"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block mixed case DROP")

        query = "SeLeCt * FrOm master..Xp_CmDsHeLl('calc')"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block mixed case xp_cmdshell")

    def test_block_benchmark_mysql(self):
        """Should block BENCHMARK() function (MySQL DoS/Timing)."""
        query = "SELECT BENCHMARK(1000000,MD5(1))"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block BENCHMARK")

    def test_block_pg_sleep(self):
        """Should block pg_sleep() function (Postgres DoS/Timing)."""
        query = "SELECT pg_sleep(10)"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block pg_sleep")

    def test_block_call(self):
        """Should block CALL statement."""
        query = "CALL dangerous_procedure()"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block CALL")

    def test_block_prepare(self):
        """Should block PREPARE statement."""
        query = "PREPARE stmt FROM 'DROP TABLE users'"
        self.assertFalse(SQLValidator.validate_read_only(query), "Failed to block PREPARE")

if __name__ == '__main__':
    unittest.main()
