"""Views & Dashboard Controller Blueprint
Provides routes for all 23 application screens, data pagination, and visualization payloads.
"""

from pathlib import Path
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_file, jsonify, abort
from flask_login import login_required, current_user
import numpy as np
import pandas as pd

from config import Config
from src.load_data import load_dataset
from src.pipeline import run_pipeline, get_cached_pipeline_results
from app.models_db import db, AlertStatus, ProjectNote
from app.auth import role_required

views_bp = Blueprint("views", __name__)

def get_context_helpers():
    """Returns general context indicators like data source and alert counts."""
    cache = get_cached_pipeline_results()
    data_source = cache.get("data_source", "Synthetic") if cache else "None"
    
    critical_count = 0
    if cache and "scored_df" in cache:
        df = cache["scored_df"]
        critical_count = int((df["risk_level"] == "Critical").sum())
        
    return {
        "data_source": data_source,
        "critical_alerts_count": critical_count
    }

@views_bp.app_context_processor
def inject_global_vars():
    return get_context_helpers()

# ==========================================
# SCREEN 1: LANDING PAGE
# ==========================================
@views_bp.route("/")
def landing():
    cache = get_cached_pipeline_results()
    return render_template("landing.html", has_data=cache is not None)

# ==========================================
# SCREEN 4: EXECUTIVE DASHBOARD
# ==========================================
@views_bp.route("/dashboard")
def dashboard():
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        return render_template("dashboard.html", has_data=False)

    df = cache["scored_df"]
    eff = cache.get("efficiency_metrics", {})
    
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

    top_flagged = []
    top_df = df.sort_values("risk_score", ascending=False).head(8)
    for _, row in top_df.iterrows():
        reasons = row.get("risk_reasons", [])
        primary = reasons[0] if len(reasons) > 0 else "Risk indicator requiring verification"
        top_flagged.append({
            "work_id": row["work_id"],
            "mp_name": row.get("mp_name", ""),
            "constituency": row.get("constituency", ""),
            "state": row.get("state", ""),
            "district": row.get("district", ""),
            "work_category": row.get("work_category", ""),
            "sanctioned_amount": row.get("sanctioned_amount", 0.0),
            "expenditure": row.get("expenditure", 0.0),
            "risk_score": row.get("risk_score", 0.0),
            "risk_level": row.get("risk_level", "Medium"),
            "primary_reason": primary
        })

    return render_template(
        "dashboard.html",
        has_data=True,
        cache_timestamp=cache.get("timestamp", ""),
        kpis=kpis,
        risk_tier_counts=risk_tier_counts,
        state_chart_data=state_chart_data,
        top_flagged_projects=top_flagged
    )

# ==========================================
# SCREEN 5: DATA INGESTION & DEMO LOADER
# ==========================================
@views_bp.route("/upload")
def upload():
    return render_template("upload.html")

@views_bp.route("/upload-file", methods=["POST"])
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
        mapped_df, metadata = load_dataset(dest_path)
        cache_result = run_pipeline(mapped_df, data_source=metadata["data_source"])
        flash(f"Successfully processed {len(mapped_df):,} records from {file.filename} in {cache_result['elapsed_seconds']}s!", "success")
        return redirect(url_for("views.dashboard"))
    except Exception as e:
        flash(f"Error parsing uploaded file: {str(e)}", "danger")
        return redirect(url_for("views.upload"))

@views_bp.route("/load-demo-data")
def load_demo_data():
    syn_file = Config.DATA_SYNTHETIC_DIR / "mplads_synthetic_5000.csv"
    if not syn_file.exists():
        from src.generate_data import generate_mplads_dataset
        df = generate_mplads_dataset(5000, 0.10)
    else:
        df = pd.read_csv(syn_file)

    cache_result = run_pipeline(df, data_source="Synthetic")
    flash(f"Benchmark demo dataset loaded successfully ({cache_result['total_projects']:,} projects evaluated in {cache_result['elapsed_seconds']}s)!", "success")
    return redirect(url_for("views.dashboard"))

# ==========================================
# SCREEN 6: VALIDATION AUDIT REPORT
# ==========================================
@views_bp.route("/validate")
def validate_view():
    cache = get_cached_pipeline_results()
    if not cache:
        return render_template("validate.html", has_data=False)
    
    val_report = cache.get("validation_report", {})
    quarantine_df = cache.get("quarantine_df", pd.DataFrame())
    
    # Sample quarantined records for display table
    quarantined_records = []
    if len(quarantine_df) > 0:
        for _, row in quarantine_df.head(25).iterrows():
            issues = row.get("validation_issues", ["Structural validation error"])
            issue_str = "; ".join(issues) if isinstance(issues, list) else str(issues)
            quarantined_records.append({
                "work_id": row.get("work_id", "N/A"),
                "state": row.get("state", "N/A"),
                "district": row.get("district", "N/A"),
                "sanctioned": row.get("sanctioned_amount", "N/A"),
                "expenditure": row.get("expenditure", "N/A"),
                "status": row.get("status", "N/A"),
                "issues": issue_str
            })

    return render_template(
        "validate.html",
        has_data=True,
        val_report=val_report,
        quarantine_count=len(quarantine_df),
        quarantined_records=quarantined_records
    )

# ==========================================
# SCREEN 7: DATA CLEANING LOGS
# ==========================================
@views_bp.route("/clean")
def clean_view():
    cache = get_cached_pipeline_results()
    if not cache:
        return render_template("clean.html", has_data=False)

    logs = cache.get("cleaning_logs", [])
    df = cache.get("scored_df", pd.DataFrame())
    preview_records = df.head(15).to_dict("records") if len(df) > 0 else []

    return render_template(
        "clean.html",
        has_data=True,
        logs=logs,
        cleaned_count=len(df),
        preview_records=preview_records
    )

# ==========================================
# SCREEN 8: DATA QUALITY SCORE
# ==========================================
@views_bp.route("/quality")
def quality_view():
    cache = get_cached_pipeline_results()
    if not cache:
        return render_template("quality.html", has_data=False)

    quality = cache.get("quality_report", {})
    return render_template("quality.html", has_data=True, quality=quality)

# ==========================================
# SCREEN 9: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
@views_bp.route("/eda")
def eda():
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        return render_template("eda.html", has_data=False)

    df = cache["scored_df"]
    
    # 1. Projects tab data
    cat_counts = df["work_category"].value_counts().to_dict()
    status_counts = df["status"].value_counts().to_dict()

    # 2. Financial tab data
    cat_fin = df.groupby("work_category").agg(
        sanctioned=("sanctioned_amount", "sum"),
        expenditure=("expenditure", "sum")
    ).round(2).to_dict("index")

    # 3. Location tab data
    state_counts = df["state"].value_counts().head(12).to_dict()

    # 4. Expenditure distribution bins
    exp_bins = [0, 10, 25, 50, 75, 100]
    exp_hist = pd.cut(df["expenditure"], bins=exp_bins, labels=["0-10L", "10-25L", "25-50L", "50-75L", "75L+"]).value_counts().sort_index().to_dict()

    return render_template(
        "eda.html",
        has_data=True,
        cat_counts=cat_counts,
        status_counts=status_counts,
        cat_fin=cat_fin,
        state_counts=state_counts,
        exp_hist=exp_hist
    )

# ==========================================
# SCREEN 10: FINANCIAL FLOW AUDIT
# ==========================================
@views_bp.route("/financial")
def financial():
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        return render_template("financial.html", has_data=False)

    df = cache["scored_df"]
    
    total_sanctioned = round(float(df["sanctioned_amount"].sum()), 2)
    total_authorized = round(float(df["authorized_amount"].sum()), 2)
    total_expenditure = round(float(df["expenditure"].sum()), 2)
    
    # By Category
    cat_flow = df.groupby("work_category").agg(
        sanctioned=("sanctioned_amount", "sum"),
        authorized=("authorized_amount", "sum"),
        expenditure=("expenditure", "sum"),
        count=("work_id", "count")
    ).round(2).reset_index().to_dict("records")

    # Top overspending projects
    overspend_df = df[df["expenditure"] > df["sanctioned_amount"]].sort_values("expenditure", ascending=False).head(10)
    overspend_list = overspend_df[["work_id", "mp_name", "work_category", "sanctioned_amount", "expenditure", "sanction_utilization"]].to_dict("records")

    return render_template(
        "financial.html",
        has_data=True,
        total_sanctioned=total_sanctioned,
        total_authorized=total_authorized,
        total_expenditure=total_expenditure,
        cat_flow=cat_flow,
        overspend_list=overspend_list
    )

# ==========================================
# SCREEN 11: EFFICIENCY & DELAYS
# ==========================================
@views_bp.route("/efficiency")
def efficiency():
    cache = get_cached_pipeline_results()
    if not cache:
        return render_template("efficiency.html", has_data=False)

    eff = cache.get("efficiency_metrics", {})
    return render_template("efficiency.html", has_data=True, eff=eff)

# ==========================================
# SCREEN 12: RULE-BASED ANOMALY LIST
# ==========================================
@views_bp.route("/rules")
def rules_view():
    cache = get_cached_pipeline_results()
    if not cache:
        return render_template("rules.html", has_data=False)

    rule_summary = cache.get("rule_summary", {})
    df = cache.get("scored_df", pd.DataFrame())
    
    # Sample records with multiple rules triggered
    multi_rule_df = df[df.get("total_rules_triggered", 0) >= 2].sort_values("total_rules_triggered", ascending=False).head(20)
    flagged_records = multi_rule_df.to_dict("records")

    return render_template(
        "rules.html",
        has_data=True,
        summary=rule_summary,
        flagged_records=flagged_records
    )

# ==========================================
# SCREEN 13: ML DETECTION SUITE & COMPARISON
# ==========================================
@views_bp.route("/ml-detection")
def ml_detection():
    cache = get_cached_pipeline_results()
    if not cache:
        return render_template("ml_detection.html", has_data=False)

    eval_results = cache.get("evaluation_results", {})
    feature_importances = cache.get("feature_importances", {})

    return render_template(
        "ml_detection.html",
        has_data=True,
        comparison_table=eval_results.get("comparison_table", []),
        metrics=eval_results.get("metrics", {}),
        feature_importances=feature_importances,
        eval_note=eval_results.get("note", "")
    )

# ==========================================
# SCREEN 14: DUPLICATE WORK DETECTION
# ==========================================
@views_bp.route("/duplicates")
def duplicates_view():
    cache = get_cached_pipeline_results()
    if not cache:
        return render_template("duplicates.html", has_data=False)

    pairs = cache.get("duplicate_pairs", [])
    return render_template("duplicates.html", has_data=True, pairs=pairs)

# ==========================================
# SCREEN 15: COST OVERRUN ANALYSIS
# ==========================================
@views_bp.route("/cost-overrun")
def cost_overrun():
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        return render_template("cost_overrun.html", has_data=False)

    df = cache["scored_df"]
    overrun_df = df[df["cost_overrun_pct"] > 0.10].sort_values("cost_overrun_pct", ascending=False)
    
    overrun_list = []
    for _, row in overrun_df.head(50).iterrows():
        overrun_list.append({
            "work_id": row["work_id"],
            "mp_name": row.get("mp_name", ""),
            "state": row.get("state", ""),
            "district": row.get("district", ""),
            "work_category": row.get("work_category", ""),
            "estimated_cost": row.get("estimated_cost", 0.0),
            "expenditure": row.get("expenditure", 0.0),
            "overrun_pct": round(row.get("cost_overrun_pct", 0.0) * 100, 1),
            "overrun_amount": round(row.get("expenditure", 0.0) - row.get("estimated_cost", 0.0), 2),
            "risk_score": row.get("risk_score", 0.0),
            "risk_level": row.get("risk_level", "Medium")
        })

    return render_template(
        "cost_overrun.html",
        has_data=True,
        total_overrun_projects=len(overrun_df),
        overrun_list=overrun_list
    )

# ==========================================
# SCREEN 16: PROJECT RISK DOSSIER (/risk/<id>)
# ==========================================
@views_bp.route("/risk/<work_id>")
def risk_detail(work_id):
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        flash("No dataset loaded.", "warning")
        return redirect(url_for("views.dashboard"))

    df = cache["scored_df"]
    match = df[df["work_id"] == work_id]
    if len(match) == 0:
        flash(f"Project '{work_id}' not found.", "danger")
        return redirect(url_for("views.risk_list"))

    project = match.iloc[0].to_dict()
    reasons = project.get("risk_reasons", ["Risk indicator requiring verification"])

    # Fetch notes from SQLite
    notes = ProjectNote.query.filter_by(work_id=work_id).order_by(ProjectNote.created_at.desc()).all()

    return render_template(
        "risk_detail.html",
        has_data=True,
        p=project,
        reasons=reasons,
        notes=notes
    )

# ==========================================
# SCREEN 17: MASTER RISK REGISTER (/risk)
# ==========================================
@views_bp.route("/risk")
def risk_list():
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        return render_template("risk_list.html", has_data=False)

    df = cache["scored_df"]
    
    # Filter parameters
    query = request.args.get("q", "").strip()
    level = request.args.get("level", "").strip()
    state = request.args.get("state", "").strip()
    category = request.args.get("category", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = 25

    filtered = df
    if query:
        q_lower = query.lower()
        filtered = filtered[
            filtered["work_id"].str.lower().str.contains(q_lower, na=False) |
            filtered["mp_name"].str.lower().str.contains(q_lower, na=False) |
            filtered["district"].str.lower().str.contains(q_lower, na=False) |
            filtered["constituency"].str.lower().str.contains(q_lower, na=False) |
            filtered["work_description"].str.lower().str.contains(q_lower, na=False)
        ]

    if level:
        filtered = filtered[filtered["risk_level"].str.lower() == level.lower()]

    if state:
        filtered = filtered[filtered["state"].str.lower() == state.lower()]

    if category:
        filtered = filtered[filtered["work_category"].str.lower() == category.lower()]

    total_filtered = len(filtered)
    # Sort descending by risk_score
    sorted_df = filtered.sort_values("risk_score", ascending=False)
    
    # Simple pagination slice
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_records = sorted_df.iloc[start_idx:end_idx].to_dict("records")
    total_pages = max(1, int(np.ceil(total_filtered / per_page)))

    # Available filter options
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
# SCREEN 18: OPERATIONAL ALERTS QUEUE
# ==========================================
@views_bp.route("/alerts")
def alerts_view():
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        return render_template("alerts.html", has_data=False)

    df = cache["scored_df"]
    # Filter high/critical or flagged works
    flagged_df = df[(df["risk_level"].isin(["High", "Critical"])) | (df.get("total_rules_triggered", 0) > 0)].sort_values("risk_score", ascending=False)

    # Load SQLite statuses
    db_statuses = {a.work_id: a.status for a in AlertStatus.query.all()}

    alerts_list = []
    for _, row in flagged_df.head(100).iterrows():
        wid = row["work_id"]
        current_status = db_statuses.get(wid, "Open")
        reasons = row.get("risk_reasons", [])
        primary = reasons[0] if len(reasons) > 0 else "Risk indicator requiring verification"

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
# SCREEN 19: COMPREHENSIVE PROJECT INVESTIGATION (/project/<id>)
# ==========================================
@views_bp.route("/project/<work_id>", methods=["GET", "POST"])
def project_detail(work_id):
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        flash("No dataset loaded.", "warning")
        return redirect(url_for("views.dashboard"))

    df = cache["scored_df"]
    match = df[df["work_id"] == work_id]
    if len(match) == 0:
        flash(f"Project '{work_id}' not found.", "danger")
        return redirect(url_for("views.risk_list"))

    project = match.iloc[0].to_dict()

    # Handle posting new audit note
    if request.method == "POST":
        note_text = request.form.get("note_text", "").strip()
        if note_text:
            author_id = current_user.id if current_user.is_authenticated else 1
            author_name = current_user.full_name if current_user.is_authenticated else "Chief Auditor"
            note = ProjectNote(
                work_id=work_id,
                author_id=author_id,
                author_name=author_name,
                note_text=note_text
            )
            db.session.add(note)
            db.session.commit()
            flash("Audit inspection note appended successfully.", "success")
            return redirect(url_for("views.project_detail", work_id=work_id))

    notes = ProjectNote.query.filter_by(work_id=work_id).order_by(ProjectNote.created_at.desc()).all()
    reasons = project.get("risk_reasons", ["Risk indicator requiring verification"])

    return render_template(
        "project_detail.html",
        has_data=True,
        p=project,
        reasons=reasons,
        notes=notes
    )

# ==========================================
# SCREEN 20: GEOGRAPHIC HOTSPOT ANALYSIS
# ==========================================
@views_bp.route("/geo")
def geo():
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        return render_template("geo.html", has_data=False)

    df = cache["scored_df"]
    
    # State summary
    state_summary = df.groupby("state").agg(
        total_projects=("work_id", "count"),
        total_sanctioned=("sanctioned_amount", "sum"),
        total_expenditure=("expenditure", "sum"),
        avg_risk=("risk_score", "mean"),
        high_risk_count=("risk_level", lambda s: s.isin(["High", "Critical"]).sum())
    ).round(2).reset_index().to_dict("records")

    # District hotspots (top 15 highest avg risk with at least 5 projects)
    dist_grp = df.groupby(["state", "district"]).agg(
        total_projects=("work_id", "count"),
        avg_risk=("risk_score", "mean"),
        critical_count=("risk_level", lambda s: (s == "Critical").sum()),
        total_spent=("expenditure", "sum")
    ).reset_index()
    dist_hotspots = dist_grp[dist_grp["total_projects"] >= 5].sort_values("avg_risk", ascending=False).head(20).round(2).to_dict("records")

    return render_template(
        "geo.html",
        has_data=True,
        state_summary=state_summary,
        dist_hotspots=dist_hotspots
    )

# ==========================================
# SCREEN 21: EMBEDDED POWER BI INTEGRATION
# ==========================================
@views_bp.route("/powerbi")
def powerbi_view():
    cache = get_cached_pipeline_results()
    has_cache = cache is not None
    powerbi_url = Config.POWERBI_URL
    return render_template("powerbi.html", has_data=has_cache, powerbi_url=powerbi_url)

# ==========================================
# SCREEN 22: SYSTEM SETTINGS (Admin Only)
# ==========================================
@views_bp.route("/settings", methods=["GET", "POST"])
def settings_view():
    if request.method == "POST":
        if current_user.is_authenticated and not current_user.is_admin():
            flash("Access denied: Settings calibration requires Administrator privileges.", "danger")
            return redirect(url_for("views.settings_view"))

        # Update weights in runtime config
        try:
            w_if = float(request.form.get("weight_if", Config.WEIGHT_ISOLATION_FOREST))
            w_svm = float(request.form.get("weight_svm", Config.WEIGHT_ONE_CLASS_SVM))
            w_rf = float(request.form.get("weight_rf", Config.WEIGHT_RANDOM_FOREST))
            w_rules = float(request.form.get("weight_rules", Config.WEIGHT_RULES))
            
            Config.WEIGHT_ISOLATION_FOREST = w_if
            Config.WEIGHT_ONE_CLASS_SVM = w_svm
            Config.WEIGHT_RANDOM_FOREST = w_rf
            Config.WEIGHT_RULES = w_rules

            new_bi_url = request.form.get("powerbi_url", "").strip()
            if new_bi_url:
                Config.POWERBI_URL = new_bi_url

            flash("Calibration parameters and weights updated successfully.", "success")
        except Exception as e:
            flash(f"Error saving settings: {e}", "danger")

    return render_template(
        "settings.html",
        weights={
            "if": Config.WEIGHT_ISOLATION_FOREST,
            "svm": Config.WEIGHT_ONE_CLASS_SVM,
            "rf": Config.WEIGHT_RANDOM_FOREST,
            "rules": Config.WEIGHT_RULES
        },
        thresholds=Config.RISK_THRESHOLDS,
        powerbi_url=Config.POWERBI_URL
    )

# ==========================================
# SCREEN 23: AUDIT REPORTS & DOWNLOADS
# ==========================================
@views_bp.route("/reports")
def reports_view():
    cache = get_cached_pipeline_results()
    return render_template("reports.html", has_data=cache is not None)

@views_bp.route("/reports/generate-pdf", methods=["POST"])
def generate_pdf_report():
    """Generates a downloadable PDF audit dossier using ReportLab."""
    cache = get_cached_pipeline_results()
    if not cache or "scored_df" not in cache:
        flash("No dataset loaded to compile report.", "warning")
        return redirect(url_for("views.reports_view"))

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors

    pdf_filename = "MPLADS_AI_Executive_Audit_Report.pdf"
    pdf_path = Config.REPORTS_DIR / pdf_filename

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Title & Header
    c.setFillColor(colors.HexColor("#0F172A"))
    c.rect(0, height - 70, width, 70, fill=1, stroke=0)
    
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(36, height - 38, "MPLADS AI - Executive Audit & Fraud Risk Report")
    
    c.setFont("Helvetica", 9)
    c.drawString(36, height - 54, "Prototype - Academic Project (B.Tech CSE Data Science) | Non-Governmental Prototype")

    # Metadata
    df = cache["scored_df"]
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
    c.drawString(36, height - 230, "2. Priority Works Requiring Verification (Sample)")
    
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
    top_crit = df[df["risk_level"] == "Critical"].head(18)
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
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.HexColor("#64748B"))
    c.drawString(36, 30, "LEGAL NOTICE: Findings are algorithmic risk indicators requiring verification; never definitive fraud determinations.")
    c.save()

    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

@views_bp.route("/reports/download-csv/<export_name>")
def download_csv(export_name):
    allowed_exports = {
        "projects_scored": Config.EXPORTS_DIR / "projects_scored.csv",
        "anomalies": Config.EXPORTS_DIR / "anomalies.csv",
        "district_summary": Config.EXPORTS_DIR / "district_summary.csv",
        "alerts": Config.EXPORTS_DIR / "alerts.csv"
    }
    target = allowed_exports.get(export_name)
    if not target or not target.exists():
        flash("Export file not found. Please refresh the pipeline first.", "warning")
        return redirect(url_for("views.reports_view"))

    return send_file(target, as_attachment=True, download_name=f"{export_name}.csv")
