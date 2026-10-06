from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_demo_credentials_and_reset_boundary():
    denied = client.post('/api/demo/auth/login', json={'email': 'user@example.com', 'password': 'bad'})
    assert denied.json()['authenticated'] is False

    accepted = client.post('/api/demo/auth/login', json={'email': 'demo@demo', 'password': 'demo'})
    assert accepted.json()['authenticated'] is True

    client.post('/api/demo/scenario/offline-camera')
    assert client.get('/api/demo/state').json()['scenario'] == 'offline-camera'

    assert client.post('/api/demo/reset').json()['scenario'] == 'online-camera'
