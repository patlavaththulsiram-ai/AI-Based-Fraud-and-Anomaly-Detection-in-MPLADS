"""Database Models for Users, Roles, Alert Triage Statuses, and Audit Notes
Persisted in SQLite via Flask-SQLAlchemy.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="analyst", nullable=False) # admin, analyst, viewer
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def is_admin(self) -> bool:
        return self.role == "admin"

    def is_analyst(self) -> bool:
        return self.role in ["admin", "analyst"]

    def can_edit(self) -> bool:
        """Viewers are read-only."""
        return self.role in ["admin", "analyst"]

class AlertStatus(db.Model):
    __tablename__ = "alert_statuses"

    id = db.Column(db.Integer, primary_key=True)
    work_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    alert_type = db.Column(db.String(100), default="Risk Indicator")
    severity = db.Column(db.String(20), default="High") # Low, Medium, High, Critical
    status = db.Column(db.String(20), default="Open") # Open, Investigating, Resolved
    assigned_to = db.Column(db.String(64), nullable=True)
    resolution_notes = db.Column(db.Text, nullable=True)
    updated_by = db.Column(db.String(64), nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProjectNote(db.Model):
    __tablename__ = "project_notes"

    id = db.Column(db.Integer, primary_key=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    author_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    author_name = db.Column(db.String(64), nullable=False)
    note_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
