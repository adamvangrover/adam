import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a hard to guess string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    CORE_INTEGRATION = True

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    CORE_INTEGRATION = False

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
