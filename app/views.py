"""Views & Dashboard Controller Blueprint
Provides routes for all application modules partitioned strictly by DatasetID.
Enforces Role-Based Access Control: Admin has full access including Data Management,
while User has access only to Dashboard, Analytics, Risk & Alerts, Reports, and Settings.
All data is queried from Microsoft SQL Server (MPLADS_AI_DB).
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    flash, session, send_file, jsonify, abort
)
from flask_login import login_required, current_user
import numpy as np
import pandas as pd

from config import Config
from app.models_db import (
    db, User, ActiveUserSession, Dataset, Project, FinancialRecord, ImplementationRecord,
    RiskScore, AnomalyResult, GROUND_TRUTH_ANOMALY_MODEL, DuplicateResult, Alert, InvestigationNote,
    SystemSetting, AuditLog, DataValidationResult, DataQualityResult
)
from app.auth import role_required, admin_required
from src.dataset_service import (
    get_available_datasets, get_dataset_by_id,
    get_dataset_pipeline_results, clear_dataset_cache
)
from src.ingest_service import ingest_dataset_to_sql_server
from src.load_data import load_dataset
from src.pipeline import run_pipeline
from src.ml_detection import evaluate_dataset

views_bp = Blueprint("views", __name__)

def get_active_dataset_data():
    """
    Retrieves the dataset payload from SQL Server for the DatasetID currently
    stored in the session. If not set, attempts to pick the primary available dataset.
    """
    dataset_id = session.get("dataset_id")
    if not dataset_id:
        datasets = get_available_datasets()
        if datasets:
            dataset_id = datasets[0].dataset_id
            session["dataset_id"] = dataset_id
            session["dataset_name"] = datasets[0].name
        else:
            return None
    return get_dataset_pipeline_results(dataset_id)

@views_bp.app_context_processor
def inject_global_vars():
    """Injects current dataset and notification indicators across templates."""
    dataset_id = session.get("dataset_id")
    data = get_active_dataset_data() if dataset_id else None

    data_source = data.get("data_source", "Audited Portal") if data else "None"
    critical_count = 0
    if data and "scored_df" in data:
        df = data["scored_df"]
        critical_count = int((df["risk_level"] == "Critical").sum())

    return {
        "data_source": data_source,
        "critical_alerts_count": critical_count,
        "current_dataset_id": dataset_id,
        "current_dataset_name": session.get("dataset_name", dataset_id or "Select Dataset")
    }

# ==========================================
# SCREEN 1: LANDING PAGE
# ==========================================
@views_bp.route("/")
def landing():
    datasets = get_available_datasets()
    return render_template("landing.html", has_data=len(datasets) > 0)

# ==========================================
# SAVED DATASET SELECTION (Both Admin & User)
# ==========================================
@views_bp.route("/datasets/select", methods=["GET", "POST"])
@login_required
def select_dataset():
    """
    Saved Dataset Selection page seen by both Admin and User after login.
    User selects a dataset and clicks Enter to set session['dataset_id'].
    """
    datasets = get_available_datasets()

    if request.method == "POST":
        selected_id = request.form.get("dataset_id", "").strip()
        ds = get_dataset_by_id(selected_id)
        if not ds:
            flash("Selected dataset does not exist in SQL Server.", "danger")
            return redirect(url_for("views.select_dataset"))

        session["dataset_id"] = ds.dataset_id
        session["dataset_name"] = ds.name

        # Log dataset switch
        try:
            db.session.add(AuditLog(
                user_id=current_user.id,
                username=current_user.username,
                action="DATASET_SELECTED",
                details=f"User switched to dataset {ds.dataset_id} ('{ds.name}').",
                ip_address=request.remote_addr
            ))
            db.session.commit()
        except Exception:
            db.session.rollback()

        flash(f"Entered workspace: '{ds.name}' ({ds.record_count} projects loaded).", "success")
        return redirect(url_for("views.dashboard"))

    return render_template("dataset_select.html", datasets=datasets)

# ==========================================
# DATASET MANAGEMENT (Admin Only)
# ==========================================
@views_bp.route("/datasets")
@admin_required
def dataset_management():
    """Admin-only overview and control of all SQL Server datasets."""
    datasets = get_available_datasets()
    return render_template("dataset_management.html", datasets=datasets)

@views_bp.route("/datasets/delete/<dataset_id>", methods=["POST"])
@admin_required
def delete_dataset(dataset_id):
    """Admin-only deletion of a dataset and its partitioned entities."""
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        flash("Dataset not found.", "warning")
        return redirect(url_for("views.dataset_management"))

    try:
        Project.query.filter_by(dataset_id=dataset_id).delete()
        FinancialRecord.query.filter_by(dataset_id=dataset_id).delete()
        ImplementationRecord.query.filter_by(dataset_id=dataset_id).delete()
        RiskScore.query.filter_by(dataset_id=dataset_id).delete()
        AnomalyResult.query.filter_by(dataset_id=dataset_id).delete()
        Alert.query.filter_by(dataset_id=dataset_id).delete()
        DuplicateResult.query.filter_by(dataset_id=dataset_id).delete()
        DataValidationResult.query.filter_by(dataset_id=dataset_id).delete()
        DataQualityResult.query.filter_by(dataset_id=dataset_id).delete()
        InvestigationNote.query.filter_by(dataset_id=dataset_id).delete()

        db.session.delete(ds)
        db.session.commit()
        clear_dataset_cache(dataset_id)

        if session.get("dataset_id") == dataset_id:
            session.pop("dataset_id", None)
            session.pop("dataset_name", None)

        flash(f"Dataset {dataset_id} and all related records purged from SQL Server.", "info")
    except Exception as e:
        db.session.rollback()
        flash(f"Error purging dataset: {e}", "danger")

    return redirect(url_for("views.dataset_management"))

# ==========================================
# SCREEN 4: EXECUTIVE DASHBOARD (Admin & User)
# ==========================================
@views_bp.route("/dashboard")
@login_required
def dashboard():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("dashboard.html", has_data=False)

    df = data["scored_df"]
    eff = data.get("efficiency_metrics", {})

    total_projects = len(df)
    total_sanctioned = float(df["sanctioned_amount"].sum())
    total_expenditure = float(df["expenditure"].sum())

    total_funds_cr = round(total_sanctioned / 100.0, 2)
    total_exp_cr = round(total_expenditure / 100.0, 2)

    status_counts = df["status"].value_counts().to_dict()
    completed = int(status_counts.get("Completed", 0))
    ongoing = int(status_counts.get("Ongoing", 0))
    delayed = int(status_counts.get("Delayed", 0))
    pending = int(status_counts.get("Pending", 0))

    high_critical_count = int(df["risk_level"].isin(["High", "Critical"]).sum())
    avg_delay = round(float(df["delay_days"].mean()), 1)
    max_delay = int(df["delay_days"].max()) if total_projects > 0 else 0

    utilization_rate = round((total_expenditure / max(1.0, total_sanctioned)) * 100.0, 1)
    avg_gap = round(float(df.get("progress_vs_spend_gap", pd.Series(0.0)).mean()) * 100.0, 1)
    avg_risk = round(float(df["risk_score"].mean()), 1)

    if avg_risk <= 39:
        avg_risk_tier = "Low"
    elif avg_risk <= 59:
        avg_risk_tier = "Medium"
    elif avg_risk <= 79:
        avg_risk_tier = "High"
    else:
        avg_risk_tier = "Critical"

    kpis = {
        "total_projects": f"{total_projects:,}",
        "total_funds_crores": f"{total_funds_cr:,.2f}",
        "total_expenditure_crores": f"{total_exp_cr:,.2f}",
        "completed_count": f"{completed:,}",
        "completed_pct": f"{(completed/max(1, total_projects))*100:.1f}",
        "ongoing_count": ongoing,
        "delayed_count": delayed,
        "pending_count": pending,
        "delayed_pending_count": f"{delayed + pending:,}",
        "high_critical_risk_count": f"{high_critical_count:,}",
        "high_critical_risk_pct": f"{(high_critical_count/max(1, total_projects))*100:.1f}",
        "avg_delay_days": avg_delay,
        "max_delay_days": max_delay,
        "utilization_rate_pct": utilization_rate,
        "avg_spend_progress_gap": avg_gap,
        "avg_risk_score": avg_risk,
        "avg_risk_tier": avg_risk_tier
    }

    risk_tier_counts = df["risk_level"].value_counts().to_dict()

    state_grp = df.groupby("state").agg(
        total=("work_id", "count"),
        risk=("risk_level", lambda s: s.isin(["High", "Critical"]).sum())
    ).sort_values("total", ascending=False).head(12)

    state_chart_data = {
        "labels": state_grp.index.tolist(),
        "total_projects": state_grp["total"].tolist(),
        "risk_projects": state_grp["risk"].tolist()
    }

    cat_grp = df.groupby("work_category")["sanctioned_amount"].sum().sort_values(ascending=False).head(6)
    category_chart_data = {
        "labels": cat_grp.index.tolist(),
        "values": [round(float(v) / 100.0, 2) for v in cat_grp.values]
    }

    top_risks = df.sort_values("risk_score", ascending=False).head(10).to_dict(orient="records")
    for r in top_risks:
        drivers = r.get("top_risk_drivers", [])
        r["primary_reason"] = drivers[0] if isinstance(drivers, list) and len(drivers) > 0 else "Multiple risk triggers"

    return render_template(
        "dashboard.html",
        has_data=True,
        kpis=kpis,
        risk_tier_counts=risk_tier_counts,
        state_chart_data=state_chart_data,
        category_chart_data=category_chart_data,
        cache_timestamp=data["cache_timestamp"],
        top_critical_projects=top_risks
    )

# ==========================================
# SCREEN 5: DATA INGESTION (Admin Only)
# ==========================================
@views_bp.route("/upload")
@admin_required
def upload():
    return render_template("upload.html")

@views_bp.route("/upload-file", methods=["POST"])
@admin_required
def upload_file():
    if "dataset_file" not in request.files:
        flash("No file part in upload request.", "danger")
        return redirect(url_for("views.upload"))

    file = request.files["dataset_file"]
    if file.filename == "":
        flash("No file selected.", "warning")
        return redirect(url_for("views.upload"))

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in Config.ALLOWED_EXTENSIONS:
        flash(f"Unsupported file type .{ext}. Please upload a CSV, XLSX, or JSON file.", "danger")
        return redirect(url_for("views.upload"))

    dest_path = Config.DATA_RAW_DIR / file.filename
    file.save(dest_path)

    try:
        ds_id = f"DS-UP-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        ds_name = f"Uploaded Audit - {Path(file.filename).stem}"
        ds_desc = f"User uploaded dataset from {file.filename} processed via AI validation and scoring engine."

        dataset, pipeline_res = ingest_dataset_to_sql_server(
            dataset_id=ds_id,
            name=ds_name,
            description=ds_desc,
            file_path=dest_path,
            uploaded_by=current_user.username
        )

        session["dataset_id"] = ds_id
        session["dataset_name"] = ds_name

        flash(f"Successfully processed and stored {len(pipeline_res['scored_df']):,} records in SQL Server MPLADS_AI_DB under {ds_id}!", "success")
        return redirect(url_for("views.validate_view"))
    except Exception as e:
        flash(f"Error parsing and ingesting file: {str(e)}", "danger")
        return redirect(url_for("views.upload"))

@views_bp.route("/load-demo-data", methods=["GET", "POST"])
@admin_required
def load_demo_data():
    if request.method == "GET":
        return render_template("load_demo.html")

    syn_file = Config.DATA_SYNTHETIC_DIR / "mplads_synthetic_500.csv"
    if not syn_file.exists():
        from src.generate_data import generate_mplads_dataset
        df = generate_mplads_dataset(500, 0.10)
        syn_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(syn_file, index=False)

    ds_id = f"DS-DEMO-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"
    ds_name = "Benchmark Multi-State Demo Audit"
    ds_desc = "Benchmark multi-state synthetic dataset evaluating 500 infrastructure projects with policy anomalies."

    ingest_dataset_to_sql_server(
        dataset_id=ds_id,
        name=ds_name,
        description=ds_desc,
        file_path=syn_file,
        uploaded_by=current_user.username
    )

    session["dataset_id"] = ds_id
    session["dataset_name"] = ds_name
    flash("Benchmark demo dataset loaded and registered in SQL Server.", "success")
    return redirect(url_for("views.dashboard"))

# ==========================================
# SCREEN 6: VALIDATION AUDIT REPORT (Admin Only)
# ==========================================
@views_bp.route("/validate")
@admin_required
def validate_view():
    data = get_active_dataset_data()
    if not data:
        return render_template("validate.html", has_data=False)

    source_report = data.get("validation_report", {})
    quarantine_count = data.get("quarantine_count", 0)
    total = data.get("total_projects", 0)
    valid_count = total
    df = data["scored_df"]

    cols_summary = []
    summary_dict = source_report.get("summary", {})
    for col_name, stats in summary_dict.items():
        if isinstance(stats, dict):
            cols_summary.append({
                "column": col_name,
                "status": stats.get("status", "valid"),
                "missing_count": stats.get("missing", 0),
                "type_errors": stats.get("type_errors", 0),
                "invalid_samples": stats.get("invalid_samples", [])
            })

    field_summary = {
        column: {
            "null_count": int(df[column].isna().sum()),
            "null_pct": round(float(df[column].isna().mean() * 100), 1)
        }
        for column in df.columns
    }
    validation_summary = {
        "total_records": total + quarantine_count,
        "valid_records": valid_count,
        "invalid_records": quarantine_count,
        "warnings": [
            f"{quarantine_count} records were quarantined."
        ] if quarantine_count else [],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_records": int(df["work_id"].duplicated().sum()),
        "invalid_dates": int(source_report.get("invalid_dates", 0)),
        "invalid_amounts": int(source_report.get("invalid_amounts", source_report.get("out_of_range_errors", 0))),
        "invalid_progress": int(source_report.get("invalid_progress", 0)),
        "field_summary": field_summary
    }

    return render_template(
        "validate.html",
        has_data=True,
        total_records=total + quarantine_count,
        valid_records=valid_count,
        quarantine_records=quarantine_count,
        quarantine_count=quarantine_count,
        val_report=validation_summary,
        columns_summary=cols_summary,
        quarantined_records=[]
    )

# ==========================================
# SCREEN 7: DATA CLEANING LOG (Admin Only)
# ==========================================
@views_bp.route("/clean")
@admin_required
def clean_view():
    data = get_active_dataset_data()
    if not data:
        return render_template("clean.html", has_data=False)

    total = data.get("total_projects", 0)
    quarantine_count = data.get("quarantine_count", 0)
    df = data["scored_df"]

    cleaning_logs = [
        {
            "timestamp": data["cache_timestamp"],
            "step": "Standardized Column Headers",
            "description": "Applied canonical schema alias mapping to match the national standard schema.",
            "impact": "Canonical project fields"
        },
        {
            "timestamp": data["cache_timestamp"],
            "step": "Date Coercion & Verification",
            "description": "Parsed sanction, target, and actual completion dates into SQL Server datetime2.",
            "impact": "Validated project dates"
        },
        {
            "timestamp": data["cache_timestamp"],
            "step": "Financial Outlay Verification",
            "description": "Validated numeric expenditure, sanctioned, and authorized fields.",
            "impact": "Checked financial values"
        },
        {
            "timestamp": data["cache_timestamp"],
            "step": "Deduplication & Partitioning",
            "description": f"Partitioned {total} records under DatasetID {session.get('dataset_id')}.",
            "impact": "Dataset partitioned"
        }
    ]

    return render_template(
        "clean.html",
        has_data=True,
        total_processed=total + quarantine_count,
        clean_count=total,
        quarantine_count=quarantine_count,
        logs=cleaning_logs,
        cleaned_count=total,
        preview_records=df.head(25).to_dict(orient="records")
    )

# ==========================================
# SCREEN 8: DATA QUALITY SCORE (Admin Only)
# ==========================================
@views_bp.route("/quality")
@admin_required
def quality_view():
    data = get_active_dataset_data()
    if not data:
        return render_template("quality.html", has_data=False)

    quality = data.get("quality_report", {})
    return render_template("quality.html", has_data=True, quality=quality)

# ==========================================
# SCREEN 9: EXPLORATORY DATA ANALYSIS (EDA) (Admin & User)
# ==========================================
@views_bp.route("/eda")
@login_required
def eda():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("eda.html", has_data=False)

    df = data["scored_df"]
    num_cols = ["sanctioned_amount", "estimated_cost", "authorized_amount", "expenditure", "progress_percent", "delay_days"]
    existing_num = [c for c in num_cols if c in df.columns]
    desc_df = df[existing_num].describe().round(2).to_dict()

    cat_dist = df["work_category"].value_counts().to_dict() if "work_category" in df.columns else {}
    status_dist = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    state_counts = df["state"].value_counts().to_dict() if "state" in df.columns else {}

    top_states = df["state"].value_counts().head(10).to_dict() if "state" in df.columns else {}
    top_agencies = df["implementing_agency"].value_counts().head(8).to_dict() if "implementing_agency" in df.columns else {}

    # Cost histogram
    hist_counts, bin_edges = np.histogram(df["sanctioned_amount"].dropna(), bins=10)
    hist_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}L" for i in range(len(hist_counts))]
    expenditure_counts, expenditure_edges = np.histogram(df["expenditure"].dropna(), bins=10)
    exp_hist = {
        f"{int(expenditure_edges[i])}-{int(expenditure_edges[i + 1])}L": int(expenditure_counts[i])
        for i in range(len(expenditure_counts))
    }

    cat_fin_frame = df.groupby("work_category").agg(
        sanctioned=("sanctioned_amount", "sum"),
        expenditure=("expenditure", "sum")
    )
    cat_fin = cat_fin_frame.to_dict(orient="index")

    corr_cols = [c for c in ["sanctioned_amount", "expenditure", "progress_percent", "delay_days", "risk_score"] if c in df.columns]
    corr_matrix = df[corr_cols].corr().round(2).to_dict() if len(corr_cols) > 1 else {}

    return render_template(
        "eda.html",
        has_data=True,
        desc_stats=desc_df,
        cat_dist=cat_dist,
        status_dist=status_dist,
        cat_counts=cat_dist,
        status_counts=status_dist,
        state_counts=state_counts,
        exp_hist=exp_hist,
        cat_fin=cat_fin,
        top_states=top_states,
        top_agencies=top_agencies,
        hist_labels=hist_labels,
        hist_counts=[int(x) for x in hist_counts],
        corr_matrix=corr_matrix,
        corr_cols=corr_cols
    )

# ==========================================
# SCREEN 10: FINANCIAL FLOW ANALYSIS (Admin & User)
# ==========================================
@views_bp.route("/financial")
@login_required
def financial():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("financial.html", has_data=False)

    df = data["scored_df"]
    total_sanctioned = float(df["sanctioned_amount"].sum())
    total_authorized = float(df.get("authorized_amount", df["sanctioned_amount"]).sum())
    total_spent = float(df["expenditure"].sum())

    total_unspent = max(0.0, total_authorized - total_spent)
    overall_utilization = round((total_spent / max(1.0, total_sanctioned)) * 100, 1)

    over_authorized_mask = df["expenditure"] > df["authorized_amount"]
    over_authorized_count = int(over_authorized_mask.sum())
    over_authorized_amount = round(float((df.loc[over_authorized_mask, "expenditure"] - df.loc[over_authorized_mask, "authorized_amount"]).sum()), 2)

    cat_fin = df.groupby("work_category").agg(
        sanctioned=("sanctioned_amount", "sum"),
        spent=("expenditure", "sum")
    ).sort_values("sanctioned", ascending=False).head(8)

    cat_fin_data = {
        "labels": cat_fin.index.tolist(),
        "sanctioned": [round(v / 100, 2) for v in cat_fin["sanctioned"].tolist()],
        "spent": [round(v / 100, 2) for v in cat_fin["spent"].tolist()]
    }
    category_flow = df.groupby("work_category").agg(
        count=("work_id", "count"),
        sanctioned=("sanctioned_amount", "sum"),
        authorized=("authorized_amount", "sum"),
        expenditure=("expenditure", "sum")
    ).reset_index().to_dict(orient="records")
    overspend_list = df[df["expenditure"] > df["sanctioned_amount"]].sort_values(
        "expenditure", ascending=False
    ).head(25).to_dict(orient="records")

    return render_template(
        "financial.html",
        has_data=True,
        total_sanctioned=total_sanctioned,
        total_authorized=total_authorized,
        total_expenditure=total_spent,
        total_spent=total_spent,
        total_unspent=total_unspent,
        overall_utilization=overall_utilization,
        over_authorized_count=over_authorized_count,
        over_authorized_amount=over_authorized_amount,
        cat_fin_data=cat_fin_data,
        cat_flow=category_flow,
        overspend_list=overspend_list
    )

# ==========================================
# SCREEN 11: IMPLEMENTATION EFFICIENCY & DELAY ANALYSIS (Admin & User)
# ==========================================
@views_bp.route("/efficiency")
@login_required
def efficiency():
    data = get_active_dataset_data()
    if not data:
        return render_template("efficiency.html", has_data=False)

    eff = data.get("efficiency_metrics", {})
    return render_template("efficiency.html", has_data=True, eff=eff)

# ==========================================
# SCREEN 12: POLICY & DETERMINISTIC RULE ENGINE (Admin & User)
# ==========================================
@views_bp.route("/rules")
@login_required
def rules_view():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("rules.html", has_data=False)

    df = data["scored_df"]
    rule_counts = data.get("rule_summary", {})

    rules_table = [
        {"id": "R01", "name": "Expenditure Exceeds Authorized Allocation", "desc": "Actual spent fund exceeds installment released by District Authority.", "count": rule_counts.get("expenditure_exceeds_authorized", 0), "severity": "High"},
        {"id": "R02", "name": "Severe Implementation Delay (> 180 Days)", "desc": "Project execution delayed more than 6 months past target date.", "count": rule_counts.get("delayed_project", 0), "severity": "High"},
        {"id": "R03", "name": "Physical vs Financial Progress Mismatch", "desc": "Financial expenditure ratio is > 30% higher than physical completion.", "count": rule_counts.get("progress_spend_mismatch", 0), "severity": "Medium"},
        {"id": "R04", "name": "Significant Cost Overrun (> 5%)", "desc": "Final expenditure exceeded initial administrative estimate.", "count": rule_counts.get("cost_overrun", 0), "severity": "Medium"},
        {"id": "R05", "name": "Duplicate Work Linguistic Similarity", "desc": "Work description matches another project in the same constituency with >= 80% similarity.", "count": rule_counts.get("high_duplicate_risk", 0), "severity": "Critical"}
    ]

    flagged_projects = df[df.get("total_rules_triggered", 0) > 0].sort_values("risk_score", ascending=False).head(30).to_dict(orient="records")
    total_projects = len(df)
    flagged_records = df[df["total_rules_triggered"] > 0]
    summary = {
        "total_audited_records": total_projects,
        "records_with_any_violation": len(flagged_records),
        "violation_rate": round((len(flagged_records) / max(1, total_projects)) * 100, 1),
        "rule_counts": {
            rule["id"]: {
                "code": rule["id"],
                "name": rule["name"],
                "severity": rule["severity"],
                "description": rule["desc"],
                "hit_count": rule["count"],
                "hit_percentage": round((rule["count"] / max(1, total_projects)) * 100, 1)
            }
            for rule in rules_table
        }
    }

    return render_template(
        "rules.html",
        has_data=True,
        summary=summary,
        flagged_records=flagged_records.sort_values("risk_score", ascending=False).head(30).to_dict(orient="records"),
        rules_table=rules_table,
        flagged_projects=flagged_projects
    )

# ==========================================
# SCREEN 13: ML ANOMALY DETECTION SUITE (Admin & User)
# ==========================================
@views_bp.route("/ml-detection")
@login_required
def ml_detection():
    dataset_id = session.get("dataset_id")
    data = get_dataset_pipeline_results(dataset_id) if dataset_id else None
    if not data or "scored_df" not in data:
        return render_template(
            "ml_detection.html",
            has_data=False,
            comparison_table=[],
            metrics={"random_forest": None},
            feature_importances={},
            chart_data=[],
            eval_note="Select a dataset with processed records to run model evaluation.",
            random_forest_status="No dataset selected.",
            top_anomalies=[],
            flagged_records=[]
        )

    df = data["scored_df"]
    ground_truth_records = AnomalyResult.query.filter_by(
        dataset_id=dataset_id,
        model_name=GROUND_TRUTH_ANOMALY_MODEL
    ).all()
    ground_truth = {
        record.work_id: int(record.is_anomaly)
        for record in ground_truth_records
    }
    evaluation = evaluate_dataset(df, ground_truth=ground_truth)

    return render_template(
        "ml_detection.html",
        has_data=True,
        eval_note=evaluation["eval_note"],
        comparison_table=evaluation["comparison_table"],
        metrics=evaluation["metrics"],
        feature_importances=evaluation["feature_importances"],
        chart_data=evaluation["chart_data"],
        random_forest_status=evaluation["random_forest_status"],
        total_projects=len(df),
        top_anomalies=evaluation["top_anomalies"],
        flagged_records=evaluation["flagged_records"]
    )

# ==========================================
# SCREEN 14: DUPLICATE WORKS DETECTION (Admin & User)
# ==========================================
@views_bp.route("/duplicates")
@login_required
def duplicates_view():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("duplicates.html", has_data=False)

    df = data["scored_df"].set_index("work_id", drop=False)
    duplicate_pairs = []
    for pair in data.get("duplicate_pairs", []):
        first = df.loc[pair["work_id_1"]]
        second = df.loc[pair["work_id_2"]]
        duplicate_pairs.append({
            "work_id_1": pair["work_id_1"],
            "work_id_2": pair["work_id_2"],
            "similarity_pct": round(float(pair["similarity"]) * 100, 1),
            "district": first["district"],
            "work_category": first["work_category"],
            "amount_1": first["sanctioned_amount"],
            "amount_2": second["sanctioned_amount"],
            "work_description_1": first["work_description"],
            "work_description_2": second["work_description"],
            "agency_1": first["implementing_agency"],
            "agency_2": second["implementing_agency"]
        })
    return render_template("duplicates.html", has_data=True, pairs=duplicate_pairs)

# ==========================================
# SCREEN 15: COST OVERRUN ANALYSIS (Admin & User)
# ==========================================
@views_bp.route("/cost-overrun")
@login_required
def cost_overrun():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("cost_overrun.html", has_data=False)

    df = data["scored_df"]
    overrun_df = df[df["cost_overrun_pct"] > 0.0].sort_values("cost_overrun_pct", ascending=False)

    total_overrun_projects = len(overrun_df)
    avg_overrun = round(float(overrun_df["cost_overrun_pct"].mean() * 100), 1) if total_overrun_projects > 0 else 0.0
    overrun_records = overrun_df.head(25).copy()
    overrun_records["overrun_amount"] = overrun_records["expenditure"] - overrun_records["estimated_cost"]
    overrun_records["overrun_pct"] = overrun_records["cost_overrun_pct"] * 100

    return render_template(
        "cost_overrun.html",
        has_data=True,
        total_overrun_projects=total_overrun_projects,
        avg_overrun_pct=avg_overrun,
        overrun_list=overrun_records.to_dict(orient="records"),
        top_overruns=overrun_records.to_dict(orient="records")
    )

# ==========================================
# SCREEN 16 & 17: MASTER RISK REGISTER & EXPLAINABILITY (Admin & User)
# ==========================================
@views_bp.route("/risk/<work_id>")
@login_required
def risk_detail(work_id):
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        flash("No dataset loaded.", "warning")
        return redirect(url_for("views.dashboard"))

    df = data["scored_df"]
    match = df[df["work_id"] == work_id]
    if len(match) == 0:
        flash(f"Project '{work_id}' not found in active dataset partition.", "danger")
        return redirect(url_for("views.risk_list"))

    project = match.iloc[0].to_dict()
    drivers = project.get("top_risk_drivers", [])

    return render_template(
        "risk_detail.html",
        has_data=True,
        p=project,
        risk_drivers=drivers
    )

@views_bp.route("/risk")
@login_required
def risk_list():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("risk_list.html", has_data=False)

    df = data["scored_df"]

    # Filter parameters
    query = request.args.get("q", "").strip().lower()
    level = request.args.get("level", "").strip()
    state = request.args.get("state", "").strip()
    category = request.args.get("category", "").strip()

    filtered = df.copy()
    if query:
        filtered = filtered[
            filtered["work_id"].str.lower().str.contains(query, na=False) |
            filtered["work_description"].str.lower().str.contains(query, na=False) |
            filtered["mp_name"].str.lower().str.contains(query, na=False)
        ]
    if level:
        filtered = filtered[filtered["risk_level"].str.lower() == level.lower()]
    if state:
        filtered = filtered[filtered["state"] == state]
    if category:
        filtered = filtered[filtered["work_category"] == category]

    # Pagination
    page = request.args.get("page", 1, type=int)
    per_page = 20
    total_filtered = len(filtered)
    total_pages = max(1, (total_filtered + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_records = filtered.sort_values("risk_score", ascending=False).iloc[start_idx:end_idx].to_dict(orient="records")

    available_states = sorted(df["state"].dropna().unique().tolist())
    available_categories = sorted(df["work_category"].dropna().unique().tolist())

    return render_template(
        "risk_list.html",
        has_data=True,
        projects=page_records,
        page=page,
        total_pages=total_pages,
        total_count=total_filtered,
        selected_query=query,
        selected_level=level,
        selected_state=state,
        selected_category=category,
        available_states=available_states,
        available_categories=available_categories
    )

# ==========================================
# SCREEN 18: OPERATIONAL ALERTS QUEUE (Admin & User)
# ==========================================
@views_bp.route("/alerts")
@login_required
def alerts_view():
    dataset_id = session.get("dataset_id")
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("alerts.html", has_data=False)

    df = data["scored_df"]
    flagged_df = df[(df["risk_level"].isin(["High", "Critical"])) | (df.get("total_rules_triggered", 0) > 0)].sort_values("risk_score", ascending=False)

    # Load SQL Server alert statuses for this dataset
    db_alerts = {a.work_id: a.status for a in Alert.query.filter_by(dataset_id=dataset_id).all()}

    alerts_list = []
    for _, row in flagged_df.head(100).iterrows():
        wid = row["work_id"]
        current_status = db_alerts.get(wid, "Open")
        drivers = row.get("top_risk_drivers", [])
        primary = drivers[0] if isinstance(drivers, list) and len(drivers) > 0 else "Risk indicator requiring official verification"

        alerts_list.append({
            "work_id": wid,
            "mp_name": row.get("mp_name", ""),
            "constituency": row.get("constituency", ""),
            "district": row.get("district", ""),
            "state": row.get("state", ""),
            "work_category": row.get("work_category", ""),
            "sanctioned_amount": row.get("sanctioned_amount", 0.0),
            "expenditure": row.get("expenditure", 0.0),
            "severity": row.get("risk_level", "Medium"),
            "risk_score": row.get("risk_score", 0.0),
            "status": current_status,
            "primary_reason": primary
        })

    return render_template("alerts.html", has_data=True, alerts=alerts_list)

# ==========================================
# SCREEN 19: PROJECT INVESTIGATION & AUDIT NOTES (Admin & User)
# ==========================================
@views_bp.route("/project/<work_id>", methods=["GET", "POST"])
@login_required
def project_detail(work_id):
    dataset_id = session.get("dataset_id")
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        flash("No dataset loaded.", "warning")
        return redirect(url_for("views.dashboard"))

    df = data["scored_df"]
    match = df[df["work_id"] == work_id]
    if len(match) == 0:
        flash(f"Project '{work_id}' not found.", "danger")
        return redirect(url_for("views.risk_list"))

    project = match.iloc[0].to_dict()

    # Append note
    if request.method == "POST":
        note_text = request.form.get("note_text", "").strip()
        if note_text:
            note = InvestigationNote(
                dataset_id=dataset_id,
                work_id=work_id,
                author_id=current_user.id,
                author_name=current_user.full_name,
                note_text=note_text
            )
            db.session.add(note)
            db.session.commit()
            flash("Audit investigation note recorded in SQL Server.", "success")
            return redirect(url_for("views.project_detail", work_id=work_id))

    notes = InvestigationNote.query.filter_by(dataset_id=dataset_id, work_id=work_id).order_by(InvestigationNote.created_at.desc()).all()
    reasons = project.get("top_risk_drivers", ["Risk indicator requiring official verification"])

    return render_template(
        "project_detail.html",
        has_data=True,
        p=project,
        reasons=reasons,
        notes=notes
    )

# ==========================================
# SCREEN 20: STATE & DISTRICT SPATIAL HOTSPOTS (Admin & User)
# ==========================================
@views_bp.route("/geo")
@login_required
def geo():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        return render_template("geo.html", has_data=False)

    df = data["scored_df"]
    state_table = df.groupby("state").agg(
        total_projects=("work_id", "count"),
        total_sanctioned=("sanctioned_amount", "sum"),
        total_expenditure=("expenditure", "sum"),
        avg_risk=("risk_score", "mean"),
        high_risk_count=("risk_level", lambda levels: levels.isin(["High", "Critical"]).sum())
    ).reset_index()

    district_table = df.groupby(["district", "state"]).agg(
        total_projects=("work_id", "count"),
        avg_risk=("risk_score", "mean"),
        critical_count=("risk_level", lambda levels: (levels == "Critical").sum()),
        total_spent=("expenditure", "sum")
    ).reset_index()
    district_hotspots = district_table[district_table["total_projects"] >= 5].sort_values(
        "avg_risk", ascending=False
    )

    return render_template(
        "geo.html",
        has_data=True,
        state_summary=state_table.sort_values("total_projects", ascending=False).to_dict(orient="records"),
        dist_hotspots=district_hotspots.to_dict(orient="records"),
        state_data=state_table.to_dict(orient="records")
    )

# ==========================================
# SCREEN 21: DATASET VISUALIZATIONS (Admin & User)
# ==========================================
@views_bp.route("/visualizations")
@login_required
def visualizations():
    data = get_active_dataset_data()
    df = data.get("scored_df") if data else None
    if df is None or df.empty:
        return render_template(
            "visualizations.html",
            has_data=False,
            dataset_id=data.get("dataset_id") if data else session.get("dataset_id")
        )

    status_counts = df["status"].fillna("Unknown").value_counts()
    risk_order = ["Low", "Medium", "High", "Critical"]
    risk_counts = df["risk_level"].fillna("Unknown").value_counts()
    risk_order.extend(level for level in risk_counts.index if level not in risk_order)

    state_summary = df.assign(
        state=df["state"].fillna("National"),
        high_risk=df["risk_level"].isin(["High", "Critical"]).astype(int)
    ).groupby("state").agg(
        projects=("work_id", "count"),
        high_risk=("high_risk", "sum"),
        expenditure=("expenditure", "sum")
    ).sort_values("projects", ascending=False).head(10)

    category_summary = df.assign(
        work_category=df["work_category"].fillna("Uncategorized")
    ).groupby("work_category").agg(
        sanctioned=("sanctioned_amount", "sum"),
        authorized=("authorized_amount", "sum"),
        expenditure=("expenditure", "sum")
    ).sort_values("sanctioned", ascending=False).head(8)

    delay_labels = ["On schedule", "1–30 days", "31–90 days", "91–180 days", "Over 180 days"]
    delays = pd.to_numeric(df["delay_days"], errors="coerce").fillna(0)
    delay_counts = [
        int((delays <= 0).sum()),
        int(delays.between(1, 30).sum()),
        int(delays.between(31, 90).sum()),
        int(delays.between(91, 180).sum()),
        int((delays > 180).sum())
    ]

    risk_driver_columns = [
        ("Financial", "financial_anomaly_score"),
        ("Schedule", "delay_anomaly_score"),
        ("ML anomaly", "ml_anomaly_score"),
        ("Duplicate", "duplicate_score"),
        ("Progress mismatch", "progress_mismatch_score"),
        ("Data quality", "quality_defect_score")
    ]
    risk_driver_labels = []
    risk_driver_values = []
    for label, column in risk_driver_columns:
        if column in df.columns:
            risk_driver_labels.append(label)
            risk_driver_values.append(round(float(pd.to_numeric(df[column], errors="coerce").fillna(0).mean()), 1))

    progress_by_status = df.assign(
        status=df["status"].fillna("Unknown"),
        progress_percent=pd.to_numeric(df["progress_percent"], errors="coerce").fillna(0)
    ).groupby("status")["progress_percent"].mean()

    total_projects = len(df)
    total_sanctioned = float(pd.to_numeric(df["sanctioned_amount"], errors="coerce").fillna(0).sum())
    total_expenditure = float(pd.to_numeric(df["expenditure"], errors="coerce").fillna(0).sum())
    total_risky = int(df["risk_level"].isin(["High", "Critical"]).sum())
    chart_data = {
        "status": {
            "labels": status_counts.index.tolist(),
            "values": [int(value) for value in status_counts.values]
        },
        "risk": {
            "labels": risk_order,
            "values": [int(risk_counts.get(level, 0)) for level in risk_order]
        },
        "regional": {
            "labels": state_summary.index.tolist(),
            "projects": [int(value) for value in state_summary["projects"].tolist()],
            "high_risk": [int(value) for value in state_summary["high_risk"].tolist()],
            "expenditure": [round(float(value) / 100, 2) for value in state_summary["expenditure"].tolist()]
        },
        "financial": {
            "labels": category_summary.index.tolist(),
            "sanctioned": [round(float(value) / 100, 2) for value in category_summary["sanctioned"].tolist()],
            "authorized": [round(float(value) / 100, 2) for value in category_summary["authorized"].tolist()],
            "expenditure": [round(float(value) / 100, 2) for value in category_summary["expenditure"].tolist()]
        },
        "delays": {"labels": delay_labels, "values": delay_counts},
        "progress": {
            "labels": progress_by_status.index.tolist(),
            "values": [round(float(value), 1) for value in progress_by_status.tolist()]
        },
        "risk_drivers": {
            "labels": risk_driver_labels,
            "values": risk_driver_values
        }
    }

    return render_template(
        "visualizations.html",
        has_data=True,
        dataset_id=data["dataset_id"],
        chart_data=chart_data,
        summary={
            "total_projects": total_projects,
            "total_sanctioned_crores": round(total_sanctioned / 100, 2),
            "total_expenditure_crores": round(total_expenditure / 100, 2),
            "high_risk_projects": total_risky
        }
    )

# ==========================================
# SCREEN 22: SYSTEM SETTINGS (Admin vs User)
# ==========================================
@views_bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings_view():
    """
    User Settings: Change Password and Logout only.
    Admin Settings: Risk Calibration, Thresholds, Change Password, Logout.
    """
    if request.method == "POST":
        form_type = request.form.get("form_type", "change_password")

        # 1. Change Password (Both User and Admin)
        if form_type == "change_password":
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not new_password or not confirm_password:
                flash("All password fields are required.", "danger")
            elif new_password != confirm_password:
                flash("New Password and Confirm Password do not match.", "danger")
            elif len(new_password) < 6:
                flash("Password must be at least 6 characters long.", "danger")
            else:
                user = db.session.get(User, current_user.id)
                user.set_password(new_password)

                db.session.add(AuditLog(
                    user_id=user.id,
                    username=user.username,
                    action="PASSWORD_CHANGE",
                    details=f"Password securely updated and hashed for user {user.username}.",
                    ip_address=request.remote_addr
                ))
                db.session.commit()
                flash("Your password has been securely updated and hashed in SQL Server.", "success")

            return redirect(url_for("views.settings_view"))

        # 2. Risk Calibration (Admin Only)
        elif form_type == "admin_config":
            if not current_user.is_admin():
                flash("Access denied: Administrative privileges required to calibrate risk models.", "danger")
                return redirect(url_for("views.settings_view"))

            try:
                w_if = float(request.form.get("weight_if", Config.WEIGHT_ISOLATION_FOREST))
                w_svm = float(request.form.get("weight_svm", Config.WEIGHT_ONE_CLASS_SVM))
                w_rf = float(request.form.get("weight_rf", Config.WEIGHT_RANDOM_FOREST))
                w_rules = float(request.form.get("weight_rules", Config.WEIGHT_RULES))

                Config.WEIGHT_ISOLATION_FOREST = w_if
                Config.WEIGHT_ONE_CLASS_SVM = w_svm
                Config.WEIGHT_RANDOM_FOREST = w_rf
                Config.WEIGHT_RULES = w_rules

                # Save in SQL Server SystemSettings
                weights_setting = SystemSetting.query.filter_by(setting_key="risk_weights").first()
                if not weights_setting:
                    weights_setting = SystemSetting(setting_key="risk_weights", setting_value="", updated_by=current_user.username)
                    db.session.add(weights_setting)
                weights_setting.setting_value = json.dumps({"if": w_if, "svm": w_svm, "rf": w_rf, "rules": w_rules})
                weights_setting.updated_at = datetime.utcnow()

                db.session.commit()
                flash("Risk calibration weights updated successfully in SQL Server.", "success")
            except Exception as e:
                db.session.rollback()
                flash(f"Error saving settings: {e}", "danger")

            return redirect(url_for("views.settings_view"))

    settings_context = {}
    if current_user.is_admin():
        active_since = datetime.utcnow() - timedelta(minutes=15)
        ActiveUserSession.query.filter(
            ActiveUserSession.last_seen < active_since
        ).delete(synchronize_session=False)
        db.session.commit()
        active_sessions = ActiveUserSession.query.filter(
            ActiveUserSession.last_seen >= active_since
        ).order_by(ActiveUserSession.last_seen.desc()).all()
        active_users_by_id = {}
        for active_session in active_sessions:
            active_users_by_id.setdefault(active_session.user_id, active_session.last_seen)
        users = User.query.order_by(User.created_at.desc()).all()
        active_user_ids = set(active_users_by_id)

        settings_context.update({
            "weights": {
                "if": Config.WEIGHT_ISOLATION_FOREST,
                "svm": Config.WEIGHT_ONE_CLASS_SVM,
                "rf": Config.WEIGHT_RANDOM_FOREST,
                "rules": Config.WEIGHT_RULES
            },
            "thresholds": Config.RISK_THRESHOLDS,
            "users": users,
            "active_user_count": len(active_user_ids),
            "active_session_count": len(active_sessions),
            "user_count": len(users),
            "admin_user_count": sum(1 for user in users if user.is_admin()),
            "regular_user_count": sum(1 for user in users if not user.is_admin()),
            "active_user_ids": active_user_ids,
        })

    return render_template("settings.html", **settings_context)

# ==========================================
# SCREEN 23: AUDIT REPORTS & DOWNLOADS (Admin & User)
# ==========================================
@views_bp.route("/reports")
@login_required
def reports_view():
    data = get_active_dataset_data()
    return render_template("reports.html", has_data=data is not None)

@views_bp.route("/reports/generate-pdf", methods=["POST"])
@login_required
def generate_pdf_report():
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        flash("No dataset loaded to compile report.", "warning")
        return redirect(url_for("views.reports_view"))

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors

    pdf_filename = f"MPLADS_AI_{session.get('dataset_id', 'Audit')}_Report.pdf"
    pdf_path = Config.REPORTS_DIR / pdf_filename

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Title & Header
    c.setFillColor(colors.HexColor("#0D1E35"))
    c.rect(0, height - 70, width, 70, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(36, height - 38, "MPLADS AI - Executive Audit & Risk Monitoring Dossier")

    c.setFont("Helvetica", 9)
    c.drawString(36, height - 54, f"Official Audit Partition: {session.get('dataset_id', 'DS-NONE')} | Generated by {current_user.full_name} ({current_user.role.title()})")

    # Metadata
    df = data["scored_df"]
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(36, height - 100, "1. Executive Summary")

    c.setFont("Helvetica", 10)
    c.drawString(36, height - 120, f"Total Projects Audited: {len(df):,}")
    c.drawString(36, height - 138, f"Total Funds Sanctioned: Rs. {df['sanctioned_amount'].sum():,.2f} Lakhs")
    c.drawString(36, height - 156, f"Total Expenditure: Rs. {df['expenditure'].sum():,.2f} Lakhs")
    c.drawString(36, height - 174, f"High & Critical Risk Works: {int(df['risk_level'].isin(['High', 'Critical']).sum()):,}")
    c.drawString(36, height - 192, f"Mean Risk Score: {df['risk_score'].mean():.1f} / 100")

    # Table of top critical works
    c.setFont("Helvetica-Bold", 12)
    c.drawString(36, height - 230, "2. Priority Works Requiring Official Verification")

    y = height - 255
    c.setFont("Helvetica-Bold", 9)
    c.drawString(36, y, "Work ID")
    c.drawString(180, y, "State / District")
    c.drawString(320, y, "Category")
    c.drawString(410, y, "Spent (L)")
    c.drawString(480, y, "Risk")
    c.drawString(530, y, "Level")
    c.line(36, y - 4, width - 36, y - 4)

    y -= 18
    c.setFont("Helvetica", 8)
    top_crit = df[df["risk_level"].isin(["Critical", "High"])].sort_values("risk_score", ascending=False).head(18)
    for _, r in top_crit.iterrows():
        c.drawString(36, y, str(r["work_id"])[:24])
        c.drawString(180, y, f"{str(r.get('state',''))[:12]} / {str(r.get('district',''))[:10]}")
        c.drawString(320, y, str(r.get("work_category", ""))[:14])
        c.drawString(410, y, f"{r.get('expenditure', 0):.1f}")
        c.drawString(480, y, f"{r.get('risk_score', 0):.1f}")
        c.drawString(530, y, str(r.get("risk_level", "")))
        y -= 14
        if y < 60:
            break

    # Compliance Footer on PDF
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(colors.HexColor("#64748B"))
    c.drawString(36, 30, "OFFICIAL NOTICE: Algorithmic risk indicators requiring verification; never definitive proof of fraud.")
    c.save()

    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

@views_bp.route("/reports/download-csv/<export_name>")
@login_required
def download_csv(export_name):
    data = get_active_dataset_data()
    if not data or "scored_df" not in data:
        flash("No active dataset loaded for CSV export.", "warning")
        return redirect(url_for("views.reports_view"))

    df = data["scored_df"]
    dataset_id = session.get("dataset_id", "DS")

    csv_path = Config.EXPORTS_DIR / f"{dataset_id}_{export_name}.csv"
    if export_name == "projects_scored":
        df.to_csv(csv_path, index=False)
    elif export_name == "anomalies":
        df[df.get("is_anomaly", False)].to_csv(csv_path, index=False)
    elif export_name == "district_summary":
        df.groupby(["state", "district"]).agg(
            total_projects=("work_id", "count"),
            total_sanctioned=("sanctioned_amount", "sum"),
            total_spent=("expenditure", "sum"),
            avg_risk=("risk_score", "mean")
        ).reset_index().to_csv(csv_path, index=False)
    elif export_name == "alerts":
        df[df["risk_level"].isin(["High", "Critical"])].to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    return send_file(csv_path, as_attachment=True, download_name=f"{dataset_id}_{export_name}.csv")
