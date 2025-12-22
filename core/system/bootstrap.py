import sys
import os
import importlib.util
import logging
from typing import List, Tuple

logger = logging.getLogger("core.system.bootstrap")

class Bootstrap:
    """
    Validates the runtime environment for Adam.
    """

    REQUIRED_PACKAGES = [
        "pandas",
        "numpy",
        "pydantic",
        "networkx",
        "langgraph",
        "textblob",
        "tweepy",
        "transformers",
        "langchain_community",
    ]

    # Packages that enable advanced features but aren't strictly required for core loop
    OPTIONAL_PACKAGES = [
        "torch",
        "flask",
        "semantic_kernel",
        "scikit_learn", # scikit-learn import name is sklearn but spec uses dist name usually, find_spec uses import name
        "yfinance"
    ]

    REQUIRED_DIRS = [
        "data",
        "logs",
        "config"
    ]

    @staticmethod
    def _get_import_name(package_name: str) -> str:
        """Maps package distribution names to import names if they differ."""
        mapping = {
            "scikit_learn": "sklearn",
            "semantic_kernel": "semantic_kernel", # same
            "python-dotenv": "dotenv"
        }
        return mapping.get(package_name, package_name)

    @staticmethod
    def check_python_version() -> bool:
        required_version = (3, 10)
        current_version = sys.version_info
        if current_version < required_version:
            logger.error(f"Python version {required_version[0]}.{required_version[1]}+ is required. Found {current_version.major}.{current_version.minor}")
            return False
        logger.info(f"Python version: {current_version.major}.{current_version.minor} (OK)")
        return True

    @staticmethod
    def check_dependencies() -> Tuple[List[str], List[str]]:
        missing_required = []
        missing_optional = []

        for pkg in Bootstrap.REQUIRED_PACKAGES:
            import_name = Bootstrap._get_import_name(pkg)
            if not importlib.util.find_spec(import_name):
                missing_required.append(pkg)

        for pkg in Bootstrap.OPTIONAL_PACKAGES:
            import_name = Bootstrap._get_import_name(pkg)
            if not importlib.util.find_spec(import_name):
                missing_optional.append(pkg)

        return missing_required, missing_optional

    @staticmethod
    def check_directories() -> List[str]:
        missing_dirs = []
        root_dir = os.getcwd()
        for d in Bootstrap.REQUIRED_DIRS:
            path = os.path.join(root_dir, d)
            if not os.path.exists(path):
                missing_dirs.append(d)
        return missing_dirs

    @classmethod
    def run(cls) -> bool:
        """
        Runs all checks. Returns True if system is ready to start (even with warnings).
        Returns False if critical failures exist.
        """
        # Configure a basic logger if not configured, to ensure output
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logger.info("Bootstrapping Adam System...")

        if not cls.check_python_version():
            return False

        missing_req, missing_opt = cls.check_dependencies()

        if missing_req:
            logger.critical(f"Missing Critical Dependencies: {', '.join(missing_req)}")
            logger.critical("Please run: pip install -r requirements.txt")
            return False

        if missing_opt:
            logger.warning(f"Missing Optional Dependencies (Capabilities reduced): {', '.join(missing_opt)}")
        else:
            logger.info("All dependencies found.")

        missing_dirs = cls.check_directories()
        if missing_dirs:
            logger.warning(f"Missing directories (creating them): {', '.join(missing_dirs)}")
            for d in missing_dirs:
                os.makedirs(d, exist_ok=True)

        logger.info("Bootstrap complete. System Ready.")
        return True
