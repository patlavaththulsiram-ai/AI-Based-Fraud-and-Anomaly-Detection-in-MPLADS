"""Unit Tests for Authentication, RBAC, Dataset Selection, and Key Routes

Verifies strict SQL Server-backed role separation and dataset session isolation.

"""



import pytest

from jinja2 import StrictUndefined

from app import create_app

from app.models_db import db, User, Dataset, Project, Alert



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



def test_landing_page_account_menu_contains_three_auth_links(client):

    """Verify the three-dot menu is the only navigation control for auth links."""

    response = client.get("/")

    assert response.status_code == 200

    assert b"MPLADS" in response.data

    assert b"User Registration" in response.data

    assert b"User Login" in response.data

    assert b"Admin Login" in response.data

    page = response.get_data(as_text=True)

    nav_start = page.index('<nav class="lp-nav">')

    nav_end = page.index("</nav>", nav_start)

    nav = page[nav_start:nav_end]

    assert "Home" not in nav

    assert "bi-three-dots-vertical" in nav

    assert 'aria-expanded="false"' in nav

    menu_start = page.index('<div class="lp-menu-items" id="landingMenuItems" hidden>')

    menu_end = page.index("</div>", menu_start) + len("</div>")

    menu = page[menu_start:menu_end]

    assert menu.count("<a ") == 3

    assert 'href="/register"' in menu

    assert 'href="/login"' in menu

    assert 'href="/admin/login"' in menu

    assert 'menuItems.hidden = true' in page

    assert 'menuItems.hidden = isOpen' in page



def test_admin_credentials_are_rejected_on_user_login(client):

    response = client.post("/login", data={

        "username": "admin",

        "password": "admin123"

    })



    assert response.status_code == 200

    assert b"Admin accounts must use Admin Login" in response.data

    with client.session_transaction() as session:

        assert "_user_id" not in session



def test_user_login_accepts_only_user_accounts(client):

    response = client.post("/login", data={

        "username": "user",

        "password": "user123"

    }, follow_redirects=False)



    assert response.status_code == 302

    assert "/datasets/select" in response.headers["Location"]

    with client.session_transaction() as session:

        assert session.get("_user_id") is not None

    admin_page_response = client.get("/datasets", follow_redirects=True)

    assert admin_page_response.status_code == 200

    assert b"Access denied" in admin_page_response.data



def test_admin_login_accepts_only_authorized_admin(client, app):

    regular_user_attempt = client.post("/admin/login", data={

        "username": "user",

        "password": "user123"

    })

    assert regular_user_attempt.status_code == 200

    assert b"Administrative credentials required" in regular_user_attempt.data

    with client.session_transaction() as session:

        assert "_user_id" not in session



    authorized_admin_attempt = client.post("/admin/login", data={

        "username": "admin@mplads-ai.gov.in",

        "password": "admin123"

    }, follow_redirects=False)

    assert authorized_admin_attempt.status_code == 302

    assert "/datasets/select" in authorized_admin_attempt.headers["Location"]

    with client.session_transaction() as session:

        authenticated_user_id = int(session["_user_id"])

    with app.app_context():

        admin_user = db.session.get(User, authenticated_user_id)

        assert admin_user is not None and admin_user.is_admin()



def test_view_templates_render_without_undefined_variables(client, app):

    """Catch missing view/template context names across the authenticated pages."""

    app.jinja_env.undefined = StrictUndefined

    with app.app_context():

        dataset = Dataset.query.filter_by(is_active=True).first()

        assert dataset is not None

        dataset_id = dataset.dataset_id

        project = Project.query.filter_by(dataset_id=dataset_id).first()

        assert project is not None

        work_id = project.work_id



    client.post("/admin/login", data={"username": "admin", "password": "admin123"})

    client.post("/datasets/select", data={"dataset_id": dataset_id})



    paths = [

        "/",

        "/datasets/select",

        "/datasets",

        "/dashboard",

        "/upload",

        "/validate",

        "/clean",

        "/quality",

        "/eda",

        "/financial",

        "/efficiency",

        "/rules",

        "/ml-detection",

        "/duplicates",

        "/cost-overrun",

        "/risk",

        f"/risk/{work_id}",

        f"/project/{work_id}",

        "/alerts",

        "/geo",

        "/visualizations",

        "/settings",

        "/reports",

    ]

    for path in paths:

        response = client.get(path)

        assert response.status_code == 200, path



def test_ml_detection_route_runs_dataset_models_and_explains_unavailable_labels(client, app):

    with app.app_context():

        dataset = Dataset.query.filter_by(is_active=True).first()

        assert dataset is not None

        dataset_id = dataset.dataset_id



    client.post("/login", data={"username": "user", "password": "user123"})

    client.post("/datasets/select", data={"dataset_id": dataset_id})

    response = client.get("/ml-detection")



    assert response.status_code == 200

    page = response.get_data(as_text=True)

    assert "Algorithmic Performance Comparison" in page

    assert "Records Evaluated" in page

    assert "Anomalies Detected" in page

    assert "potential anomalies requiring verification" in page.casefold()

    assert "Ground-truth labels are unavailable" in page



def test_user_login_page_renders(client):

    """Verify user login page renders with Username/Email and Password."""

    response = client.get("/login")

    assert response.status_code == 200

    assert b"User Sign In" in response.data

    assert b"Username or Email" in response.data

    assert b"Forgot Password" in response.data



def test_admin_login_page_renders(client):

    """Verify separate admin login page renders with Admin Username and Password."""

    response = client.get("/admin/login")

    assert response.status_code == 200

    assert b"Authorized Admin Terminal" in response.data

    assert b"Admin Username" in response.data

    assert b"Admin Login" in response.data



def test_visualizations_show_dataset_tabs_without_external_embeds(client):

    client.post("/login", data={"username": "user", "password": "user123"})

    response = client.get("/visualizations")



    assert response.status_code == 200

    for dashboard in (

        b"Visualizations",

        b"Executive Dashboard",

        b"Financial Dashboard",

        b"Efficiency Dashboard",

        b"AI Risk Dashboard",

        b"Regional Dashboard",

    ):

        assert dashboard in response.data



    page = response.get_data(as_text=True).casefold()

    assert "power bi" not in page

    assert "<iframe" not in page

    assert "chart.umd.min.js" in page

    assert "visualizationdata" in page

    with client.session_transaction() as session:

        assert session["dataset_id"].casefold() in page



def test_dashboard_hides_advisory_and_dataset_switch_for_all_roles(client, app):

    client.post("/login", data={"username": "user", "password": "user123"})

    user_dashboard = client.get("/dashboard")



    assert user_dashboard.status_code == 200

    assert b"Official Verification Advisory" not in user_dashboard.data

    assert b"Switch Monitored Dataset" not in user_dashboard.data

    assert b">Dataset:</span>" not in user_dashboard.data

    assert b"Switch</span>" not in user_dashboard.data



    with client.session_transaction() as session:

        session.clear()

    client.post("/admin/login", data={"username": "admin", "password": "admin123"})

    admin_dashboard = client.get("/dashboard")

    assert admin_dashboard.status_code == 200

    assert b"Official Verification Advisory" not in admin_dashboard.data

    assert b"Switch Monitored Dataset" not in admin_dashboard.data

    assert b">Dataset:</span>" not in admin_dashboard.data

    assert b"Switch</span>" not in admin_dashboard.data



def test_admin_settings_lists_active_users(client, app):

    user_client = app.test_client()

    user_client.post("/login", data={"username": "user", "password": "user123"})

    user_client.get("/dashboard")



    client.post("/admin/login", data={"username": "admin", "password": "admin123"})

    admin_settings = client.get("/settings")



    assert admin_settings.status_code == 200

    assert b"User Management / Active Users" in admin_settings.data

    assert b"Active Users:" in admin_settings.data

    assert b"Total Users:" in admin_settings.data

    assert b"Audit Monitoring Officer" in admin_settings.data

    assert b"Chief Project Auditor" in admin_settings.data



def test_settings_remove_external_dashboard_configuration_for_all_roles(client):

    client.post("/login", data={"username": "user", "password": "user123"})

    user_settings = client.get("/settings")

    assert user_settings.status_code == 200

    assert b"Power BI" not in user_settings.data

    assert b'name="powerbi_url"' not in user_settings.data



    with client.session_transaction() as session:

        session.clear()

    client.post("/admin/login", data={"username": "admin", "password": "admin123"})

    admin_settings = client.get("/settings")

    assert admin_settings.status_code == 200

    assert b"Power BI" not in admin_settings.data

    assert b'name="powerbi_url"' not in admin_settings.data

    assert b"Risk Score Calibration" in admin_settings.data



def test_prototype_label_is_removed_from_application_pages(client):

    for path in ("/", "/login", "/register", "/admin/login"):

        response = client.get(path)

        assert response.status_code == 200

        page = response.get_data(as_text=True).casefold()

        assert "academic prototype" not in page

        assert "b.tech cse data science" not in page

        assert "non-official prototype" not in page



def test_user_registration_flow(client, app):

    """Verify public registration creates account with User role only (no role selection)."""

    with app.app_context():

        # Remove test user if previously existed

        User.query.filter_by(username="test_auditor_unique").delete()

        db.session.commit()



    res_reg = client.post("/register", data={

        "full_name": "Test Field Auditor",

        "username": "test_auditor_unique",

        "email": "auditor_unique@mplads-ai.gov.in",

        "password": "auditorpassword123",

        "confirm_password": "auditorpassword123",

        "role": "admin"  # Malicious attempt to self-assign admin

    }, follow_redirects=True)

    assert res_reg.status_code == 200



    with app.app_context():

        u = User.query.filter_by(username="test_auditor_unique").first()

        assert u is not None

        # Must strictly be assigned 'user' role

        assert u.role == "user"

        assert u.is_user()

        assert not u.is_admin()



def test_admin_login_success(client):

    """Verify authorized Admin credentials log into admin portal and redirect to dataset selection."""

    res = client.post("/admin/login", data={

        "username": "admin",

        "password": "admin123"

    }, follow_redirects=False)

    assert res.status_code == 302

    assert "/datasets/select" in res.headers["Location"]



def test_user_cannot_login_to_admin_portal(client):

    """Verify regular user account is rejected on admin login portal."""

    res = client.post("/admin/login", data={

        "username": "user",

        "password": "user123"

    }, follow_redirects=True)

    assert res.status_code == 200

    assert b"Access denied: Administrative credentials required" in res.data



def test_user_cannot_access_data_management(client):

    """Verify standard user is blocked from data management routes."""

    # Log in as user

    client.post("/login", data={"username": "user", "password": "user123"})



    # Try accessing Admin Data Management routes

    admin_routes = ["/datasets", "/upload", "/validate", "/clean", "/quality"]

    for r in admin_routes:

        res = client.get(r, follow_redirects=True)

        assert b"Access denied" in res.data or b"Dashboard" in res.data



def test_admin_can_access_data_management(client):

    """Verify Admin can access data management modules."""

    client.post("/admin/login", data={"username": "admin", "password": "admin123"})

    res = client.get("/datasets")

    assert res.status_code == 200

    assert b"Dataset Management" in res.data



def test_dataset_selection_sets_session(client):

    """Verify selecting a dataset sets the active partition in session."""

    client.post("/login", data={"username": "user", "password": "user123"})

    res = client.post("/datasets/select", data={"dataset_id": "DS-2024-NAT-001"}, follow_redirects=False)

    assert res.status_code == 302

    assert "/dashboard" in res.headers["Location"]



def test_api_routes_require_login(client):

    """API endpoints should reject anonymous users with a 401 JSON response."""

    endpoints = [

        "/api/summary",

        "/api/scores?limit=10",

        "/api/alerts",

        "/api/project/TEST-W-001"

    ]

    for route in endpoints:

        response = client.get(route)

        assert response.status_code == 401

        assert response.is_json

        assert response.get_json()["error"] == "Authentication required"