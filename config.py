import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "mplads-ai-sentinel-secret-key-2026-audit")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR / 'mplads_audit.db'}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Storage Directories
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
    DATA_SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
    DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_STORE_DIR = BASE_DIR / "models_store"
    EXPORTS_DIR = BASE_DIR / "exports"
    REPORTS_DIR = BASE_DIR / "reports"
    NOTEBOOKS_DIR = BASE_DIR / "notebooks"
    CACHE_DIR = BASE_DIR / "data" / "processed" / "cache"

    # Upload constraints
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json"}

    # Random reproducibility
    RANDOM_STATE = 42

    # Default Risk Weights (sum to 1.0)
    WEIGHT_ISOLATION_FOREST = 0.25
    WEIGHT_ONE_CLASS_SVM = 0.20
    WEIGHT_RANDOM_FOREST = 0.35
    WEIGHT_RULES = 0.20

    # Risk Score Classification Thresholds
    RISK_THRESHOLDS = {
        "LOW": (0, 39),
        "MEDIUM": (40, 59),
        "HIGH": (60, 79),
        "CRITICAL": (80, 100)
    }

    # Power BI Configuration
    # Fallback to an explanatory dashboard view or user-provided workspace report
    POWERBI_URL = os.environ.get(
        "POWERBI_URL",
        "https://app.powerbi.com/view?r=eyJrIjoiMDY4ZDFiYzEtYTU2MS00OGFkLWE5YjktYTM4NzAxZjE0MDlkIiwidCI6IjY2ZWYzOTc2LTMzMTQtNDBiZS04Y2E4LWQxODJkMjIzODlmNyJ9"
    )

    # Canonical Column Names
    CANONICAL_COLUMNS = [
        "work_id",
        "mp_name",
        "constituency",
        "state",
        "district",
        "implementing_agency",
        "work_category",
        "work_description",
        "sanctioned_amount",
        "estimated_cost",
        "authorized_amount",
        "expenditure",
        "sanction_date",
        "expected_completion_date",
        "actual_completion_date",
        "status",
        "progress_percent",
        "latitude",
        "longitude"
    ]

    # Flexible Mapping from Real MPLADS/Government Portals to Canonical Schema
    COLUMN_ALIASES = {
        "work_id": ["work_id", "workid", "project_id", "projectid", "id", "work_code", "scheme_code"],
        "mp_name": ["mp_name", "mp", "member_of_parliament", "honble_mp", "name_of_mp", "parliamentarian"],
        "constituency": ["constituency", "pc_name", "parliamentary_constituency", "lok_sabha_constituency", "constituency_name"],
        "state": ["state", "state_name", "state_ut", "st_name"],
        "district": ["district", "district_name", "dist_name", "district_hq"],
        "implementing_agency": ["implementing_agency", "agency", "executing_agency", "ia", "department", "nodal_agency"],
        "work_category": ["work_category", "category", "sector", "work_type", "type_of_work", "head_of_expenditure"],
        "work_description": ["work_description", "description", "work_name", "name_of_work", "details", "title"],
        "sanctioned_amount": ["sanctioned_amount", "sanction_amount", "amount_sanctioned", "sanctioned_cost", "sanction_amt"],
        "estimated_cost": ["estimated_cost", "estimate_cost", "estimated_amount", "estimate_amount", "project_cost"],
        "authorized_amount": ["authorized_amount", "amount_released", "released_amount", "authorized_cost", "installment_released"],
        "expenditure": ["expenditure", "expenditure_incurred", "actual_expenditure", "spent_amount", "amount_spent", "funds_utilized"],
        "sanction_date": ["sanction_date", "date_of_sanction", "approval_date", "admin_sanction_date", "start_sanction"],
        "expected_completion_date": ["expected_completion_date", "target_date", "scheduled_completion", "expected_end_date", "target_completion"],
        "actual_completion_date": ["actual_completion_date", "date_of_completion", "actual_end_date", "completed_on", "completion_date"],
        "status": ["status", "work_status", "project_status", "current_status", "physical_status"],
        "progress_percent": ["progress_percent", "physical_progress", "progress_pct", "percentage_physical_progress", "progress"],
        "latitude": ["latitude", "lat", "geo_lat", "location_lat"],
        "longitude": ["longitude", "long", "lng", "geo_lon", "location_long"]
    }

# Ensure all vital project directories exist
for folder in [
    Config.DATA_RAW_DIR,
    Config.DATA_SYNTHETIC_DIR,
    Config.DATA_PROCESSED_DIR,
    Config.MODELS_STORE_DIR,
    Config.EXPORTS_DIR,
    Config.REPORTS_DIR,
    Config.NOTEBOOKS_DIR,
    Config.CACHE_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)
