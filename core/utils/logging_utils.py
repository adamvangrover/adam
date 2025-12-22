import logging
import logging.config
import os
import yaml


def setup_logging(config=None, default_path='config/logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.info(f"Logging configured from {path}")
            except Exception as e:
                print(f"Error loading logging config: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        logging.info("Logging configured with default basicConfig")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    """
    return logging.getLogger(name)

class MilestoneLogger(logging.LoggerAdapter):
    """
    Logger adapter to enforce consistent milestone logging.
    """
    def milestone(self, msg, *args, **kwargs):
        self.info(f"✅ Milestone: {msg}", *args, **kwargs)

def get_milestone_logger(name: str) -> MilestoneLogger:
    """
    Get a MilestoneLogger instance.
    """
    logger = logging.getLogger(name)
    return MilestoneLogger(logger, {})
