with open("tests/test_cyclical_agents.py", "r") as f:
    content = f.read()

content = content.replace("            result = result.model_dump()", "                result = result.model_dump()")

with open("tests/test_cyclical_agents.py", "w") as f:
    f.write(content)
