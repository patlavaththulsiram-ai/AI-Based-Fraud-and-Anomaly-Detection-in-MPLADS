"""JSON REST API Endpoints
Provides programmatic access to project risk scores, operational alerts,
system KPI summaries, and project detail records strictly partitioned by DatasetID.
Persisted in Microsoft SQL Server (MPLADS_AI_DB).
"""

from flask import Blueprint, jsonify, request, session
from src.dataset_service import get_dataset_pipeline_results, get_available_datasets
from app.models_db import db, Alert, User
from flask_login import current_user, login_required
import pandas as pd
import numpy as np

api_bp = Blueprint("api", __name__, url_prefix="/api")

def _get_api_data():
    dataset_id = request.args.get("dataset_id") or session.get("dataset_id")
    if not dataset_id:
        ds_list = get_available_datasets()
        if ds_list:
            dataset_id = ds_list[0].dataset_id
            session["dataset_id"] = dataset_id
            session["dataset_name"] = ds_list[0].name
        else:
            return None
    return get_dataset_pipeline_results(dataset_id)

@api_bp.route("/summary", methods=["GET"])
@login_required
def api_summary():
    data = _get_api_data()
    if not data or "scored_df" not in data:
        return jsonify({"error": "No dataset loaded", "has_data": False}), 404

    df = data["scored_df"]
    total = len(df)
    total_sanctioned = float(df["sanctioned_amount"].sum())
    total_expenditure = float(df["expenditure"].sum())
    critical_count = int((df["risk_level"] == "Critical").sum())
    high_count = int((df["risk_level"] == "High").sum())

    return jsonify({
        "status": "success",
        "dataset_id": data.get("dataset_id"),
        "dataset_name": data.get("dataset_name"),
        "data_source": data.get("data_source", "Audited Portal"),
        "total_projects": total,
        "total_sanctioned_lakhs": round(total_sanctioned, 2),
        "total_expenditure_lakhs": round(total_expenditure, 2),
        "utilization_rate_pct": round((total_expenditure / max(1.0, total_sanctioned)) * 100.0, 1),
        "avg_risk_score": round(float(df["risk_score"].mean()), 1),
        "critical_risk_projects": critical_count,
        "high_risk_projects": high_count,
        "risk_tier_counts": df["risk_level"].value_counts().to_dict()
    })

@api_bp.route("/scores", methods=["GET"])
@login_required
def api_scores():
    data = _get_api_data()
    if not data or "scored_df" not in data:
        return jsonify({"error": "No dataset loaded"}), 404

    df = data["scored_df"]
    limit = min(request.args.get("limit", 100, type=int), 1000)
    level = request.args.get("level")

    filtered = df
    if level:
        filtered = filtered[filtered["risk_level"].str.lower() == level.lower()]

    records = []
    for _, row in filtered.head(limit).iterrows():
        records.append({
            "dataset_id": data.get("dataset_id"),
            "work_id": row["work_id"],
            "mp_name": row.get("mp_name", ""),
            "state": row.get("state", ""),
            "district": row.get("district", ""),
            "work_category": row.get("work_category", ""),
            "sanctioned_amount": row.get("sanctioned_amount", 0.0),
            "expenditure": row.get("expenditure", 0.0),
            "risk_score": row.get("risk_score", 0.0),
            "risk_level": row.get("risk_level", "Medium"),
            "duplicate_risk_flag": int(row.get("duplicate_score", 0) >= 0.70),
            "rules_triggered": int(row.get("total_rules_triggered", 0))
        })

    return jsonify({
        "status": "success",
        "dataset_id": data.get("dataset_id"),
        "count": len(records),
        "total_available": len(filtered),
        "data": records
    })

@api_bp.route("/alerts", methods=["GET"])
@login_required
def api_alerts():
    data = _get_api_data()
    if not data or "scored_df" not in data:
        return jsonify({"error": "No dataset loaded"}), 404

    df = data["scored_df"]
    dataset_id = data.get("dataset_id")
    high_critical = df[df["risk_level"].isin(["High", "Critical"])].sort_values("risk_score", ascending=False)
    
    # Merge status from SQL Server
    db_statuses = {a.work_id: a.status for a in Alert.query.filter_by(dataset_id=dataset_id).all()}

    alerts_list = []
    for _, row in high_critical.iterrows():
        wid = row["work_id"]
        status = db_statuses.get(wid, "Open")
        drivers = row.get("top_risk_drivers", [])
        primary = drivers[0] if isinstance(drivers, list) and len(drivers) > 0 else "Risk indicator requiring official verification"

        alerts_list.append({
            "dataset_id": dataset_id,
            "work_id": wid,
            "mp_name": row.get("mp_name", ""),
            "district": row.get("district", ""),
            "state": row.get("state", ""),
            "severity": row.get("risk_level"),
            "risk_score": row.get("risk_score"),
            "status": status,
            "primary_reason": primary
        })

    return jsonify({
        "status": "success",
        "dataset_id": dataset_id,
        "total_alerts": len(alerts_list),
        "alerts": alerts_list
    })

@api_bp.route("/alerts/update", methods=["POST"])
@login_required
def api_alerts_update():
    """Allows updating an alert status: Open -> Investigating -> Resolved."""
    data = request.get_json() or {}
    work_id = data.get("work_id")
    new_status = data.get("status")
    dataset_id = data.get("dataset_id") or session.get("dataset_id", "DS-2024-NAT-001")

    if not work_id or not new_status:
        return jsonify({"success": False, "message": "Missing work_id or status"}), 400

    if new_status not in ["Open", "Investigating", "Resolved"]:
        return jsonify({"success": False, "message": "Invalid status option"}), 400

    alert = Alert.query.filter_by(dataset_id=dataset_id, work_id=work_id).first()
    updater_name = current_user.username if current_user.is_authenticated else "auditor"

    if alert:
        alert.status = new_status
        alert.assigned_to = updater_name
    else:
        alert = Alert(
            dataset_id=dataset_id,
            work_id=work_id,
            status=new_status,
            assigned_to=updater_name
        )
        db.session.add(alert)

    db.session.commit()
    return jsonify({"success": True, "work_id": work_id, "status": new_status})

@api_bp.route("/project/<work_id>", methods=["GET"])
@login_required
def api_project_detail(work_id):
    data = _get_api_data()
    if not data or "scored_df" not in data:
        return jsonify({"error": "No dataset loaded"}), 404

    df = data["scored_df"]
    match = df[df["work_id"] == work_id]
    if len(match) == 0:
        return jsonify({"error": f"Project '{work_id}' not found in active dataset"}), 404

    row = match.iloc[0].to_dict()
    clean_row = {}
    for k, v in row.items():
        if pd.isna(v):
            clean_row[k] = None
        elif isinstance(v, (np.integer, int)):
            clean_row[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            clean_row[k] = float(v)
        elif isinstance(v, (datetime, pd.Timestamp)):
            clean_row[k] = str(v)
        else:
            clean_row[k] = v

    return jsonify({"status": "success", "project": clean_row})
