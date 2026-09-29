"""Authentication & Role-Based Access Control Blueprint
Manages user sessions, registration, login, and authorization guards (admin, analyst, viewer).
"""

from functools import wraps
from flask import Blueprint, render_template, redirect, url_for, flash, request, abort
from flask_login import login_user, logout_user, login_required, current_user
from app.models_db import db, User

auth_bp = Blueprint("auth", __name__)

def role_required(*allowed_roles):
    """Decorator to enforce role permissions (e.g. admin, analyst)."""
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if current_user.role not in allowed_roles:
                flash(f"Access denied: This action requires one of these roles: {', '.join(allowed_roles)}.", "danger")
                return redirect(url_for("views.dashboard"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))

    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.full_name}! Signed in as {user.role.title()}.", "success")
            next_page = request.args.get("next")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            return redirect(url_for("views.dashboard"))
        else:
            flash("Invalid username/email or password.", "danger")

    return render_template("login.html")

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
        role = request.form.get("role", "analyst")

        if not full_name or not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("Username is already taken.", "warning")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("Email address is already registered.", "warning")
            return render_template("register.html")

        if role not in ["admin", "analyst", "viewer"]:
            role = "viewer"

        new_user = User(
            full_name=full_name,
            username=username,
            email=email,
            role=role
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! You may now log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("views.landing"))

def seed_initial_users():
    """Seeds default demo accounts if no users exist."""
    if User.query.count() == 0:
        demo_users = [
            {"username": "admin", "email": "admin@mplads-ai.gov.in", "full_name": "Chief Project Auditor", "role": "admin", "pass": "admin123"},
            {"username": "analyst", "email": "analyst@mplads-ai.gov.in", "full_name": "Senior Data Analyst", "role": "analyst", "pass": "analyst123"},
            {"username": "viewer", "email": "viewer@mplads-ai.gov.in", "full_name": "Public Oversight Observer", "role": "viewer", "pass": "viewer123"}
        ]
        for u in demo_users:
            user = User(username=u["username"], email=u["email"], full_name=u["full_name"], role=u["role"])
            user.set_password(u["pass"])
            db.session.add(user)
        db.session.commit()
        print("Default demo user accounts initialized: admin, analyst, viewer.")
