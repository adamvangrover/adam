from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    # Filter out comments, flags (start with -), and empty lines
    reqs = []
    for line in lineiter:
        if line and not line.startswith("#"):
            # Clean up line: remove flags like --index-url
            # If line starts with -, skip it (flags only line)
            if line.startswith("-"):
                continue

            # Split by space and take the first part (package==version)
            # This handles 'package==1.0 --index-url ...'
            parts = line.split()
            if parts:
                req = parts[0]
                if not req.startswith("-"):
                    reqs.append(req)
    return reqs

# Read requirements.txt
try:
    install_requires = parse_requirements('requirements.txt')
except FileNotFoundError:
    install_requires = []

setup(
    name='adam-project',
    version='23.5.0',
    description='Adam: Autonomous Financial Analyst Agent',
    author='Adam Team',
    packages=find_packages(include=['core', 'core.*', 'services', 'services.*']),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'adam=core.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
    ],
)
