import re

content = ""
with open("tests/test_system_logger.py", "r") as f:
    content = f.read()

replacement = """
def test_system_logger_init(temp_log_file):
    logger = SystemLogger(log_file=str(temp_log_file))
    assert logger.log_file == str(temp_log_file)
"""

content = re.sub(
    r"def test_system_logger_init\(temp_log_file\):.*?(?=\ndef test_system_logger_log_event)",
    replacement,
    content,
    flags=re.DOTALL
)

with open("tests/test_system_logger.py", "w") as f:
    f.write(content)
