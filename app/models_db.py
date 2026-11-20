"""Database Models for MPLADS AI Monitoring System
Persisted in Microsoft SQL Server (MPLADS_AI_DB) via Flask-SQLAlchemy & PyODBC.
All analysis and project tables are partitioned and isolated by DatasetID.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()
GROUND_TRUTH_ANOMALY_MODEL = "Ground_Truth"

# 1. Users
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="user", nullable=False)  # "admin" or "user"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def is_admin(self) -> bool:
        return self.role.lower() == "admin"

    def is_user(self) -> bool:
        return self.role.lower() == "user"

    def can_manage_data(self) -> bool:
        return self.is_admin()

# Active login sessions are refreshed during authenticated requests. A session is
# considered active for the Admin user list when seen within the last 15 minutes.
class ActiveUserSession(db.Model):
    __tablename__ = "active_user_sessions"

    session_id = db.Column(db.String(128), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    last_seen = db.Column(db.DateTime, nullable=False, index=True)

# 2. Datasets
class Dataset(db.Model):
    __tablename__ = "datasets"

    dataset_id = db.Column(db.String(64), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.String(500), nullable=True)
    record_count = db.Column(db.Integer, default=0)
    uploaded_by = db.Column(db.String(64), default="System")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    file_path = db.Column(db.String(500), nullable=True)
    data_source = db.Column(db.String(100), default="Synthetic/Audited")

# 3. MPs
class MP(db.Model):
    __tablename__ = "mps"

    mp_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    mp_name = db.Column(db.String(120), nullable=False)
    constituency = db.Column(db.String(120), nullable=True)
    state = db.Column(db.String(100), nullable=True)

# 4. Districts
class District(db.Model):
    __tablename__ = "districts"

    district_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    district_name = db.Column(db.String(120), nullable=False)
    state = db.Column(db.String(100), nullable=True)

# 5. Agencies
class Agency(db.Model):
    __tablename__ = "agencies"

    agency_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    agency_name = db.Column(db.String(200), nullable=False)

# 6. Projects
class Project(db.Model):
    __tablename__ = "projects"

    project_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    mp_name = db.Column(db.String(120), nullable=True)
    constituency = db.Column(db.String(120), nullable=True)
    state = db.Column(db.String(100), nullable=True)
    district = db.Column(db.String(120), nullable=True)
    implementing_agency = db.Column(db.String(200), nullable=True)
    work_category = db.Column(db.String(100), nullable=True)
    work_description = db.Column(db.Text, nullable=True)
    sanctioned_amount = db.Column(db.Float, default=0.0)
    estimated_cost = db.Column(db.Float, default=0.0)
    authorized_amount = db.Column(db.Float, default=0.0)
    expenditure = db.Column(db.Float, default=0.0)
    sanction_date = db.Column(db.DateTime, nullable=True)
    expected_completion_date = db.Column(db.DateTime, nullable=True)
    actual_completion_date = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), default="Ongoing")
    progress_percent = db.Column(db.Float, default=0.0)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    delay_days = db.Column(db.Integer, default=0)
    cost_overrun_pct = db.Column(db.Float, default=0.0)
    progress_vs_spend_gap = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 7. FinancialRecords
class FinancialRecord(db.Model):
    __tablename__ = "financial_records"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    sanctioned_amount = db.Column(db.Float, default=0.0)
    estimated_cost = db.Column(db.Float, default=0.0)
    authorized_amount = db.Column(db.Float, default=0.0)
    expenditure = db.Column(db.Float, default=0.0)
    utilization_rate = db.Column(db.Float, default=0.0)
    unspent_balance = db.Column(db.Float, default=0.0)
    excess_spend = db.Column(db.Float, default=0.0)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)

# 8. ImplementationRecords
class ImplementationRecord(db.Model):
    __tablename__ = "implementation_records"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    status = db.Column(db.String(50), default="Ongoing")
    progress_percent = db.Column(db.Float, default=0.0)
    delay_days = db.Column(db.Integer, default=0)
    sanction_date = db.Column(db.DateTime, nullable=True)
    expected_completion_date = db.Column(db.DateTime, nullable=True)
    actual_completion_date = db.Column(db.DateTime, nullable=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)

# 9. DataUploads
class DataUpload(db.Model):
    __tablename__ = "data_uploads"

    upload_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_by = db.Column(db.String(64), default="admin")
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    row_count = db.Column(db.Integer, default=0)
    column_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(50), default="Success")

# 10. DataValidationResults
class DataValidationResult(db.Model):
    __tablename__ = "data_validation_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    total_records = db.Column(db.Integer, default=0)
    valid_records = db.Column(db.Integer, default=0)
    invalid_records = db.Column(db.Integer, default=0)
    missing_critical_fields = db.Column(db.Integer, default=0)
    type_errors = db.Column(db.Integer, default=0)
    out_of_range_errors = db.Column(db.Integer, default=0)
    validation_details_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 11. DataQualityResults
class DataQualityResult(db.Model):
    __tablename__ = "data_quality_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    completeness_score = db.Column(db.Float, default=0.0)
    validity_score = db.Column(db.Float, default=0.0)
    consistency_score = db.Column(db.Float, default=0.0)
    uniqueness_score = db.Column(db.Float, default=0.0)
    overall_quality_score = db.Column(db.Float, default=0.0)
    summary_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 12. AnomalyResults
class AnomalyResult(db.Model):
    __tablename__ = "anomaly_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    model_name = db.Column(db.String(50), nullable=False)
    anomaly_score = db.Column(db.Float, default=0.0)
    is_anomaly = db.Column(db.Boolean, default=False)
    reasons_json = db.Column(db.Text, nullable=True)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)

# 13. DuplicateResults
class DuplicateResult(db.Model):
    __tablename__ = "duplicate_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id_1 = db.Column(db.String(100), nullable=False)
    work_id_2 = db.Column(db.String(100), nullable=False)
    similarity_score = db.Column(db.Float, default=0.0)
    matching_criteria = db.Column(db.String(255), nullable=True)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)

# 14. RiskScores
class RiskScore(db.Model):
    __tablename__ = "risk_scores"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    risk_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(20), default="Low")
    financial_anomaly_score = db.Column(db.Float, default=0.0)
    delay_anomaly_score = db.Column(db.Float, default=0.0)
    ml_anomaly_score = db.Column(db.Float, default=0.0)
    duplicate_score = db.Column(db.Float, default=0.0)
    progress_mismatch_score = db.Column(db.Float, default=0.0)
    quality_defect_score = db.Column(db.Float, default=0.0)
    top_risk_drivers_json = db.Column(db.Text, nullable=True)
    calculated_at = db.Column(db.DateTime, default=datetime.utcnow)

# 15. Alerts
class Alert(db.Model):
    __tablename__ = "alerts"

    alert_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    alert_type = db.Column(db.String(100), default="Risk Indicator")
    severity = db.Column(db.String(20), default="High")  # Low, Medium, High, Critical
    status = db.Column(db.String(20), default="Open")    # Open, Investigating, Resolved
    description = db.Column(db.Text, nullable=True)
    assigned_to = db.Column(db.String(64), nullable=True)
    resolution_notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 16. InvestigationNotes
class InvestigationNote(db.Model):
    __tablename__ = "investigation_notes"

    note_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.String(64), nullable=False, index=True)
    work_id = db.Column(db.String(100), nullable=False, index=True)
    author_id = db.Column(db.Integer, nullable=False)
    author_name = db.Column(db.String(64), nullable=False)
    note_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 17. AuditLogs
class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=True)
    username = db.Column(db.String(64), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# 18. SystemSettings
class SystemSetting(db.Model):
    __tablename__ = "system_settings"

    setting_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    setting_key = db.Column(db.String(100), unique=True, nullable=False)
    setting_value = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(db.String(64), nullable=True)

# Backward-compatibility aliases for templates and views
AlertStatus = Alert
ProjectNote = InvestigationNote
