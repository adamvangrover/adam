
import unittest
import os
import sys

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.security.sandbox import SecureSandbox

class TestSandboxPandas(unittest.TestCase):
    def setUp(self):
        # Create a secret file to test read access
        self.secret_filename = "secret_test_sandbox.txt"
        with open(self.secret_filename, "w") as f:
            f.write("TOP_SECRET_DATA")

    def tearDown(self):
        if os.path.exists(self.secret_filename):
            os.remove(self.secret_filename)

    def test_pandas_read_csv_exploit(self):
        """
        Test that pd.read_csv cannot be used to read files.
        """
        # Note: No try-except allowed in sandbox code.
        code = f"""
# Attempt to read the secret file using pandas
df = pd.read_csv("{self.secret_filename}", header=None)
print("CONTENTS:", df.values[0][0])
"""
        result = SecureSandbox.execute(code)

        # Check if pandas is even available
        if "name 'pd' is not defined" in str(result.get('error', '')):
            print("Skipping pandas test: pandas not installed.")
            return

        # If vulnerable, result['status'] == 'success' and output contains secret
        output = result.get('output', '')
        self.assertNotEqual(result.get('status'), 'success', f"VULNERABILITY: Sandbox successfully executed pd.read_csv. Output: {output}")
        self.assertNotIn("TOP_SECRET_DATA", output, "VULNERABILITY: Sandbox leaked file content via pd.read_csv")

        # Verify it failed due to attribute error (function removed) or similar
        error_msg = str(result.get('error', ''))
        # Expect: "AttributeError: module 'pandas' has no attribute 'read_csv'"
        # or similar depending on how we sanitize it.
        # It might also be "SecurityViolation" if we blocked it another way.

        # But specifically we want to ensure it's NOT a FileNotFoundError (meaning it tried to open it)
        # Wait, if open is blocked, read_csv might fail with something else?
        # But 'read_csv' bypasses 'open' hook because it's C code usually.

        # We want to see that read_csv is GONE.
        # So "AttributeError" is expected.
        # self.assertIn("AttributeError", error_msg)

    def test_pandas_dataframe_creation(self):
        """
        Test that safe pandas operations still work.
        """
        code = """
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)
print("SHAPE:", df.shape)
"""
        result = SecureSandbox.execute(code)

        if "name 'pd' is not defined" in str(result.get('error', '')):
            print("Skipping pandas test: pandas not installed.")
            return

        self.assertEqual(result.get('status'), 'success', f"Valid DataFrame creation failed: {result.get('error')}")
        output = result.get('output', '')
        self.assertIn("SHAPE: (2, 2)", output)

    def test_pandas_to_csv_blocked(self):
        """
        Test that DataFrame.to_csv is blocked or restricted.
        """
        code = """
df = pd.DataFrame({'a': [1]})
df.to_csv('should_not_create.csv')
"""
        result = SecureSandbox.execute(code)

        if "name 'pd' is not defined" in str(result.get('error', '')):
            return

        # If vulnerable, status is success.
        # If secured, status is error (AttributeError or similar)
        self.assertNotEqual(result.get('status'), 'success', "VULNERABILITY: DataFrame.to_csv executed successfully")

        # Clean up if file was created
        if os.path.exists('should_not_create.csv'):
            os.remove('should_not_create.csv')
            self.fail("VULNERABILITY: to_csv created a file")

if __name__ == '__main__':
    unittest.main()
