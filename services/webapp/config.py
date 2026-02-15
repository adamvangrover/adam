import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///adam.db'
    CORE_INTEGRATION = True
    CORS_ALLOWED_ORIGINS = []

    # 🛡️ Sentinel: DoS Protection
    # Limit maximum request body size to 16MB to prevent memory exhaustion attacks.
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    @staticmethod
    def init_app(app):
        if not app.config.get('SECRET_KEY'):
            if app.config.get('DEBUG') or app.config.get('TESTING'):
                app.config['SECRET_KEY'] = 'dev-secret-key-change-me'
                import logging
                logging.warning("SECRET_KEY not set. Using insecure default for development.")
            else:
                raise RuntimeError(
                    "SECRET_KEY environment variable is not set! "
                    "This is required for production security."
                )


class DevelopmentConfig(Config):
    DEBUG = True
    CORS_ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    CORE_INTEGRATION = False


class ProductionConfig(Config):
    """
    🛡️ Sentinel: Production configuration.
    - DEBUG is False
    - Requires SECRET_KEY to be set
    - Logs to stdout/stderr for container orchestration
    """
    DEBUG = False
    TESTING = False

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Log to stderr
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}
