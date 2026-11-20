"""Authentication & Role-Based Access Control Blueprint
Manages user authentication, registration, single-authorized admin verification,
and strict role-based access control (Admin vs User).
"""

from functools import wraps
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from app.models_db import ActiveUserSession, db, User, AuditLog

auth_bp = Blueprint("auth", __name__)

def role_required(*allowed_roles):
    """Enforces role-based permissions (admin or user)."""
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            user_role = (current_user.role or "").lower()
            allowed = [r.lower() for r in allowed_roles]
            if user_role not in allowed:
                flash(f"Access denied: This module requires {' or '.join(allowed_roles).title()} authorization.", "danger")
                return redirect(url_for("views.dashboard"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def admin_required(f):
    """Specific decorator for Admin-only modules."""
    return role_required("admin")(f)

# ==========================================
# USER LOGIN
# ==========================================
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("views.select_dataset"))

    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter(
            User.role == "user",
            (User.username == identifier) | (User.email == identifier)
        ).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()

            # Log sign-in
            try:
                db.session.add(AuditLog(
                    user_id=user.id,
                    username=user.username,
                    action="USER_LOGIN",
                    details=f"User {user.username} ({user.role}) logged in successfully.",
                    ip_address=request.remote_addr
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()

            flash(f"Welcome back, {user.full_name}! Please select an active dataset to proceed.", "success")

            # Both Admin and User must first see Saved Dataset Selection page
            return redirect(url_for("views.select_dataset"))
        else:
            flash("Invalid User username/email or password. Admin accounts must use Admin Login.", "danger")

    return render_template("login.html")

# ==========================================
# ADMIN LOGIN (COMPLETELY SEPARATE)
# ==========================================
@auth_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if current_user.is_authenticated:
        if current_user.is_admin():
            return redirect(url_for("views.select_dataset"))
        else:
            flash("You are currently signed in as a standard User. Sign in with Admin credentials to access the Admin terminal.", "warning")
            logout_user()

    if request.method == "POST":
        admin_username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter(
            (User.username == admin_username) | (User.email == admin_username)
        ).first()

        if user and user.check_password(password):
            if not user.is_admin() or user.username != "admin":
                flash("Access denied: Administrative credentials required. Regular user accounts cannot access the Admin Portal.", "danger")
                return render_template("admin_login.html")

            login_user(user)
            user.last_login = datetime.utcnow()

            # Log Admin sign-in
            try:
                db.session.add(AuditLog(
                    user_id=user.id,
                    username=user.username,
                    action="ADMIN_LOGIN",
                    details=f"Authorized Admin {user.username} entered the Administrative Terminal.",
                    ip_address=request.remote_addr
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()

            flash(f"Authorized Admin Access Granted. Welcome, {user.full_name}!", "success")
            return redirect(url_for("views.select_dataset"))
        else:
            flash("Invalid Admin username or password.", "danger")

    return render_template("admin_login.html")

# ==========================================
# USER REGISTRATION
# Public registration ONLY creates User accounts.
# No role selection allowed. Users cannot become Admin.
# ==========================================
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not full_name or not username or not email or not password or not confirm:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("Username is already taken. Please choose another.", "warning")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("Email address is already registered.", "warning")
            return render_template("register.html")

        # PUBLIC REGISTRATION ALWAYS CREATES USER ROLE ONLY
        new_user = User(
            full_name=full_name,
            username=username,
            email=email,
            role="user"
        )
        new_user.set_password(password)
        db.session.add(new_user)

        # Audit log
        db.session.add(AuditLog(
            username=username,
            action="USER_REGISTRATION",
            details=f"Public registration: User account '{username}' created with role 'user'.",
            ip_address=request.remote_addr
        ))
        db.session.commit()

        flash("Registration successful! Your User account has been created. Please sign in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")

# ==========================================
# FORGOT PASSWORD
# ==========================================
@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        flash("If your registered email exists in the system, password reset instructions have been forwarded to your nodal department.", "info")
        return redirect(url_for("auth.login"))
    return render_template("forgot_password.html")

# ==========================================
# LOGOUT
# ==========================================
@auth_bp.route("/logout")
@login_required
def logout():
    username = current_user.username
    session.pop("dataset_id", None)
    session.pop("dataset_name", None)
    active_session_id = session.pop("active_user_session_id", None)

    try:
        if active_session_id:
            active_session = db.session.get(ActiveUserSession, active_session_id)
            if active_session:
                db.session.delete(active_session)
        db.session.add(AuditLog(
            user_id=current_user.id,
            username=username,
            action="USER_LOGOUT",
            details=f"User {username} signed out.",
            ip_address=request.remote_addr
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()

    logout_user()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("views.landing"))

def seed_initial_users():
    """
    Seeds the single authorized Admin account and default demo User account.
    Enforces that only one authorized Admin account exists.
    """
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(
            username="admin",
            email="admin@mplads-ai.gov.in",
            full_name="Chief Project Auditor",
            role="admin"
        )
        admin.set_password("admin123")
        db.session.add(admin)
        print("Authorized Admin account initialized: admin / admin123")
    else:
        admin.role = "admin"

    user = User.query.filter_by(username="user").first()
    if not user:
        user = User(
            username="user",
            email="user@mplads-ai.gov.in",
            full_name="Audit Monitoring Officer",
            role="user"
        )
        user.set_password("user123")
        db.session.add(user)
        print("Standard User account initialized: user / user123")
    else:
        user.role = "user"

    # Ensure any legacy analyst/viewer accounts are mapped to user role
    legacy_users = User.query.filter(User.username.notin_(["admin", "user"])).all()
    for u in legacy_users:
        if u.role != "user":
            u.role = "user"

    db.session.commit()
