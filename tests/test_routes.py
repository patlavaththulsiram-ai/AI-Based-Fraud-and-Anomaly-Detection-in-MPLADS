"""Unit Tests for Stage 4: Flask App Factory, Authentication, and Key Routes
"""

import pytest
from app import create_app
from app.models_db import db, User, AlertStatus

@pytest.fixture
def app():
    test_app = create_app()
    test_app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False
    })
    yield test_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_landing_page(client):
    """Verify landing page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"MPLADS AI" in response.data
    assert b"Prototype - Academic Project (B.Tech CSE Data Science)" in response.data

def test_login_page(client):
    """Verify login page renders with authentication form."""
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Auditor Sign In" in response.data

def test_registration_and_login_flow(client, app):
    """Verify user registration and authentication logic."""
    # Register new user
    res_reg = client.post("/register", data={
        "full_name": "Test Auditor",
        "username": "test_auditor",
        "email": "test@audit.in",
        "password": "testpassword123",
        "confirm_password": "testpassword123",
        "role": "analyst"
    }, follow_redirects=True)
    assert res_reg.status_code == 200

    # Login with valid credentials
    res_login = client.post("/login", data={
        "username": "test_auditor",
        "password": "testpassword123"
    }, follow_redirects=True)
    assert res_login.status_code == 200
    assert b"Welcome back, Test Auditor" in res_login.data

def test_dashboard_route(client):
    """Verify executive dashboard renders with KPI cards."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Executive Audit Dashboard" in response.data

def test_upload_route(client):
    """Verify upload ingestion portal renders with dropzone."""
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Data Ingestion Portal" in response.data

def test_api_summary_and_scores(client):
    """Verify REST API endpoints return valid JSON responses."""
    res_sum = client.get("/api/summary")
    assert res_sum.status_code in [200, 404] # 200 if cache exists, 404 if empty
    if res_sum.status_code == 200:
        data = res_sum.get_json()
        assert data["status"] == "success"
        assert "total_projects" in data

    res_scores = client.get("/api/scores?limit=10")
    assert res_scores.status_code in [200, 404]
    if res_scores.status_code == 200:
        sdata = res_scores.get_json()
        assert sdata["status"] == "success"
        assert len(sdata["data"]) <= 10

def test_api_alert_update(client, app):
    """Verify operational alert status update via POST API."""
    res = client.post("/api/alerts/update", json={
        "work_id": "TEST-W-001",
        "status": "Investigating"
    })
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    assert json_data["status"] == "Investigating"

    # Verify status in database
    with app.app_context():
        alert = AlertStatus.query.filter_by(work_id="TEST-W-001").first()
        assert alert is not None
        assert alert.status == "Investigating"
