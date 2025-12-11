import pytest
import sys
import os

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.api.server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_cors_restricted(client):
    """
    Test that CORS rejects arbitrary origins.
    """
    response = client.get('/api/state', headers={'Origin': 'http://evil.com'})
    acao = response.headers.get('Access-Control-Allow-Origin')
    # Should be None or not equal to evil.com
    # Flask-Cors returns None if origin is not allowed
    assert acao != 'http://evil.com' and acao != '*', f"CORS not restricted: {acao}"

def test_cors_allowed(client):
    """
    Test that CORS allows localhost.
    """
    response = client.get('/api/state', headers={'Origin': 'http://localhost:3000'})
    acao = response.headers.get('Access-Control-Allow-Origin')
    assert acao == 'http://localhost:3000', f"CORS failed for allowed origin: {acao}"
