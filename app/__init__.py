"""Flask Application Factory & Blueprint Registration
"""

import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, redirect, request, session, url_for
from flask_login import LoginManager, current_user
from config import Config
from app.models_db import ActiveUserSession, db, User

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to access this auditing module."
login_manager.login_message_category = "info"

@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/api/"):
        return jsonify({"error": "Authentication required"}), 401
    return redirect(url_for("auth.login", next=request.path))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def track_active_user():
    if not current_user.is_authenticated:
        return

    now = datetime.utcnow()
    session_id = session.get("active_user_session_id")
    active_session = (
        db.session.get(ActiveUserSession, session_id)
        if session_id
        else None
    )

    if active_session is None:
        session_id = secrets.token_urlsafe(32)
        session["active_user_session_id"] = session_id
        db.session.add(ActiveUserSession(
            session_id=session_id,
            user_id=current_user.id,
            last_seen=now
        ))
        db.session.commit()
    elif active_session.last_seen < now - timedelta(minutes=5):
        active_session.last_seen = now
        db.session.commit()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    app.before_request(track_active_user)

    # Register blueprints
    from app.auth import auth_bp, seed_initial_users
    from app.views import views_bp
    from app.api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        db.create_all()
        seed_initial_users()

    return app
