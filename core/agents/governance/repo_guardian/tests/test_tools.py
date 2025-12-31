import unittest
from core.agents.governance.repo_guardian.tools import SecurityScanner, StaticAnalyzer

class TestSecurityScanner(unittest.TestCase):
    def setUp(self):
        self.scanner = SecurityScanner()

    def test_scan_content_no_secrets(self):
        content = "def hello(): print('hello')"
        findings = self.scanner.scan_content(content)
        self.assertEqual(len(findings), 0)

    def test_scan_content_aws_key(self):
        # Fake AWS key pattern
        content = "aws_key = 'AKIA1234567890123456'"
        findings = self.scanner.scan_content(content)
        # Expect 2 findings: one specific AWS key match, and one generic assignment match
        self.assertEqual(len(findings), 2)

        types = [f['type'] for f in findings]
        self.assertIn('aws_access_key', types)
        self.assertIn('generic_secret', types)

        aws_finding = next(f for f in findings if f['type'] == 'aws_access_key')
        self.assertTrue('***' in aws_finding['snippet'])

    def test_scan_content_generic_secret(self):
        content = "api_token = 'super_secret_value'"
        findings = self.scanner.scan_content(content)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['type'], 'generic_secret')

class TestStaticAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = StaticAnalyzer()

    def test_count_loc(self):
        content = "line1\nline2\nline3"
        self.assertEqual(self.analyzer.count_loc(content), 3)

    def test_analyze_python_code_valid(self):
        code = """
import os

def my_func(a: int) -> int:
    '''Docs.'''
    return a + 1
"""
        report = self.analyzer.analyze_python_code(code)
        self.assertEqual(len(report['missing_docstrings']), 0)
        self.assertEqual(len(report['missing_type_hints']), 0)
        self.assertEqual(len(report['dangerous_functions']), 0)
        self.assertIn('os', report['imports'])
        self.assertIn('my_func', report['functions'])

    def test_analyze_python_code_issues(self):
        code = """
def bad_func(x):
    return x
"""
        report = self.analyzer.analyze_python_code(code)
        self.assertEqual(len(report['missing_docstrings']), 1)
        self.assertEqual(len(report['missing_type_hints']), 2) # Return + Arg

    def test_analyze_dangerous_functions(self):
        code = """
import os
os.system('rm -rf /')
eval('1+1')
"""
        report = self.analyzer.analyze_python_code(code)
        self.assertEqual(len(report['dangerous_functions']), 2)

if __name__ == '__main__':
    unittest.main()
