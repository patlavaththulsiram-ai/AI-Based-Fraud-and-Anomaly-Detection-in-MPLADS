"""Flask Application Factory & Blueprint Registration
"""

from pathlib import Path
from flask import Flask
from flask_login import LoginManager
from config import Config
from app.models_db import db, User

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to access this auditing module."
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)

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
