"""Dataset Ingestion & SQL Server Persistence Service

Parses uploaded CSV/XLSX datasets, executes validation, cleaning, quality evaluation,

feature engineering, ML anomaly scoring, duplicate work detection, composite risk ranking,

and commits partitioned records to Microsoft SQL Server (MPLADS_AI_DB).

"""



import json

from pathlib import Path

from datetime import datetime

from typing import Dict, Any, Tuple, Optional

import pandas as pd

import numpy as np



from app.models_db import (

    db, Dataset, MP, District, Agency, Project,

    FinancialRecord, ImplementationRecord, DataUpload,

    DataValidationResult, DataQualityResult, AnomalyResult,

    DuplicateResult, RiskScore, Alert, SystemSetting, AuditLog,

    GROUND_TRUTH_ANOMALY_MODEL

)

from src.load_data import load_dataset

from src.pipeline import run_pipeline

from src.dataset_service import clear_dataset_cache



def _ground_truth_label(value):

    """Return a binary source label when the input is explicitly valid."""

    if pd.isna(value):

        return None

    if isinstance(value, (bool, np.bool_)):

        return int(value)

    try:

        numeric_value = float(value)

    except (TypeError, ValueError):

        normalized = str(value).strip().lower()

        if normalized in {"true", "yes", "1"}:

            return 1

        if normalized in {"false", "no", "0"}:

            return 0

        return None

    return int(numeric_value) if numeric_value in (0, 1) else None





def to_sql_datetime(val):

    if val is None or pd.isna(val):

        return None

    try:

        dt = pd.to_datetime(val, errors='coerce')

        if pd.isna(dt) or dt is pd.NaT:

            return None

        py_dt = dt.to_pydatetime()

        if py_dt.year < 1900 or py_dt.year > 2100:

            return None

        return py_dt

    except Exception:

        return None



def ingest_dataset_to_sql_server(

    dataset_id: str,

    name: str,

    description: str,

    file_path: Path,

    uploaded_by: str = "admin"

) -> Tuple[Dataset, Dict[str, Any]]:

    """

    Executes full pipeline and writes partitioned records directly to SQL Server MPLADS_AI_DB.

    """

    # Clean any pre-existing records for dataset_id

    existing_ds = Dataset.query.filter_by(dataset_id=dataset_id).first()

    if existing_ds:

        Project.query.filter_by(dataset_id=dataset_id).delete()

        FinancialRecord.query.filter_by(dataset_id=dataset_id).delete()

        ImplementationRecord.query.filter_by(dataset_id=dataset_id).delete()

        RiskScore.query.filter_by(dataset_id=dataset_id).delete()

        AnomalyResult.query.filter_by(dataset_id=dataset_id).delete()

        Alert.query.filter_by(dataset_id=dataset_id).delete()

        DuplicateResult.query.filter_by(dataset_id=dataset_id).delete()

        DataValidationResult.query.filter_by(dataset_id=dataset_id).delete()

        DataQualityResult.query.filter_by(dataset_id=dataset_id).delete()

        DataUpload.query.filter_by(dataset_id=dataset_id).delete()

        MP.query.filter_by(dataset_id=dataset_id).delete()

        District.query.filter_by(dataset_id=dataset_id).delete()

        Agency.query.filter_by(dataset_id=dataset_id).delete()

        db.session.delete(existing_ds)

        db.session.commit()



    # Load canonical data

    raw_df, meta = load_dataset(file_path)



    # Run complete ML pipeline

    pipeline_res = run_pipeline(raw_df, data_source=meta.get("data_source", "Uploaded"), force_retrain=True)

    scored_df = pipeline_res["scored_df"]

    val_rep = pipeline_res["validation_report"]

    qual_rep = pipeline_res["quality_report"]

    dup_pairs = pipeline_res["duplicate_pairs"]



    # 1. Insert Dataset

    dataset = Dataset(

        dataset_id=dataset_id,

        name=name,

        description=description,

        record_count=len(scored_df),

        uploaded_by=uploaded_by,

        created_at=datetime.utcnow(),

        is_active=True,

        file_path=str(file_path),

        data_source=meta.get("file_type", "CSV").upper()

    )

    db.session.add(dataset)

    db.session.flush()



    # 2. Insert DataUpload

    upload_rec = DataUpload(

        dataset_id=dataset_id,

        filename=file_path.name,

        uploaded_by=uploaded_by,

        upload_date=datetime.utcnow(),

        row_count=len(raw_df),

        column_count=len(raw_df.columns),

        status="Success"

    )

    db.session.add(upload_rec)



    # 3. Insert DataValidationResult

    val_rec = DataValidationResult(

        dataset_id=dataset_id,

        total_records=val_rep.get("total_records", len(raw_df)),

        valid_records=val_rep.get("valid_records", len(scored_df)),

        invalid_records=val_rep.get("invalid_records", 0),

        missing_critical_fields=val_rep.get("missing_critical_fields", 0),

        type_errors=val_rep.get("type_errors", 0),

        out_of_range_errors=val_rep.get("out_of_range_errors", 0),

        validation_details_json=json.dumps(val_rep.get("summary", {})),

        created_at=datetime.utcnow()

    )

    db.session.add(val_rec)



    # 4. Insert DataQualityResult

    qual_rec = DataQualityResult(

        dataset_id=dataset_id,

        completeness_score=float(qual_rep.get("completeness", 95.0)),

        validity_score=float(qual_rep.get("validity", 92.0)),

        consistency_score=float(qual_rep.get("consistency", 94.0)),

        uniqueness_score=float(qual_rep.get("uniqueness", 98.0)),

        overall_quality_score=float(qual_rep.get("overall_score", 94.8)),

        summary_json=json.dumps(qual_rep),

        created_at=datetime.utcnow()

    )

    db.session.add(qual_rec)



    # 5. Insert Dimensions: MPs, Districts, Agencies

    unique_mps = scored_df[["mp_name", "constituency", "state"]].dropna(subset=["mp_name"]).drop_duplicates()

    for _, row in unique_mps.iterrows():

        db.session.add(MP(

            dataset_id=dataset_id,

            mp_name=str(row["mp_name"])[:120],

            constituency=str(row.get("constituency", ""))[:120],

            state=str(row.get("state", ""))[:100]

        ))



    unique_districts = scored_df[["district", "state"]].dropna(subset=["district"]).drop_duplicates()

    for _, row in unique_districts.iterrows():

        db.session.add(District(

            dataset_id=dataset_id,

            district_name=str(row["district"])[:120],

            state=str(row.get("state", ""))[:100]

        ))



    unique_agencies = scored_df["implementing_agency"].dropna().unique()

    for ag in unique_agencies:

        db.session.add(Agency(

            dataset_id=dataset_id,

            agency_name=str(ag)[:200]

        ))



    db.session.flush()



    # 6. Insert Projects, FinancialRecords, ImplementationRecords, RiskScores, AnomalyResults, Alerts

    for idx, row in scored_df.iterrows():

        w_id = str(row["work_id"])

        s_date = to_sql_datetime(row.get("sanction_date"))

        e_date = to_sql_datetime(row.get("expected_completion_date"))

        a_date = to_sql_datetime(row.get("actual_completion_date"))



        sanc_amt = float(row.get("sanctioned_amount", 0.0) or 0.0)

        est_cost = float(row.get("estimated_cost", sanc_amt) or sanc_amt)

        auth_amt = float(row.get("authorized_amount", sanc_amt) or sanc_amt)

        exp = float(row.get("expenditure", 0.0) or 0.0)

        prog_pct = float(row.get("progress_percent", 0.0) or 0.0)

        delay = int(row.get("delay_days", 0) or 0)

        cost_overrun = float(row.get("cost_overrun_pct", 0.0) or 0.0)

        prog_gap = float(row.get("progress_vs_spend_gap", 0.0) or 0.0)



        lat = float(row["latitude"]) if pd.notnull(row.get("latitude")) else None

        lng = float(row["longitude"]) if pd.notnull(row.get("longitude")) else None



        proj = Project(

            dataset_id=dataset_id,

            work_id=w_id,

            mp_name=str(row.get("mp_name", ""))[:120],

            constituency=str(row.get("constituency", ""))[:120],

            state=str(row.get("state", ""))[:100],

            district=str(row.get("district", ""))[:120],

            implementing_agency=str(row.get("implementing_agency", ""))[:200],

            work_category=str(row.get("work_category", ""))[:100],

            work_description=str(row.get("work_description", "")),

            sanctioned_amount=sanc_amt,

            estimated_cost=est_cost,

            authorized_amount=auth_amt,

            expenditure=exp,

            sanction_date=s_date,

            expected_completion_date=e_date,

            actual_completion_date=a_date,

            status=str(row.get("status", "Ongoing"))[:50],

            progress_percent=prog_pct,

            latitude=lat,

            longitude=lng,

            delay_days=delay,

            cost_overrun_pct=cost_overrun,

            progress_vs_spend_gap=prog_gap

        )

        db.session.add(proj)



        util = round((exp / max(0.01, sanc_amt)) * 100.0, 2)

        unspent = max(0.0, auth_amt - exp)

        excess = max(0.0, exp - auth_amt)

        fin_rec = FinancialRecord(

            dataset_id=dataset_id,

            work_id=w_id,

            sanctioned_amount=sanc_amt,

            estimated_cost=est_cost,

            authorized_amount=auth_amt,

            expenditure=exp,

            utilization_rate=util,

            unspent_balance=unspent,

            excess_spend=excess

        )

        db.session.add(fin_rec)



        imp_rec = ImplementationRecord(

            dataset_id=dataset_id,

            work_id=w_id,

            status=str(row.get("status", "Ongoing"))[:50],

            progress_percent=prog_pct,

            delay_days=delay,

            sanction_date=s_date,

            expected_completion_date=e_date,

            actual_completion_date=a_date

        )

        db.session.add(imp_rec)



        r_score = float(row.get("risk_score", 0.0) or 0.0)

        r_tier = str(row.get("risk_level", "Low"))

        drivers = row.get("top_risk_drivers", [])

        if isinstance(drivers, (list, np.ndarray)):

            drivers_json = json.dumps(list(drivers))

        else:

            drivers_json = json.dumps([str(drivers)])



        risk_rec = RiskScore(

            dataset_id=dataset_id,

            work_id=w_id,

            risk_score=r_score,

            risk_level=r_tier,

            financial_anomaly_score=float(row.get("financial_anomaly_score", 0.0) or 0.0),

            delay_anomaly_score=float(row.get("delay_anomaly_score", 0.0) or 0.0),

            ml_anomaly_score=float(row.get("ml_anomaly_score", 0.0) or 0.0),

            duplicate_score=float(row.get("duplicate_score", 0.0) or 0.0),

            progress_mismatch_score=float(row.get("progress_mismatch_score", 0.0) or 0.0),

            quality_defect_score=float(row.get("quality_defect_score", 0.0) or 0.0),

            top_risk_drivers_json=drivers_json

        )

        db.session.add(risk_rec)



        is_anom = bool(r_score >= 60.0 or row.get("total_rules_triggered", 0) > 0)

        anom_rec = AnomalyResult(

            dataset_id=dataset_id,

            work_id=w_id,

            model_name="Ensemble_Risk_Engine",

            anomaly_score=r_score,

            is_anomaly=is_anom,

            reasons_json=drivers_json

        )

        db.session.add(anom_rec)



        source_label = _ground_truth_label(row.get("is_anomaly"))

        if source_label is not None:

            db.session.add(AnomalyResult(

                dataset_id=dataset_id,

                work_id=w_id,

                model_name=GROUND_TRUTH_ANOMALY_MODEL,

                anomaly_score=float(source_label),

                is_anomaly=bool(source_label),

                reasons_json=None

            ))



        if r_tier in ["Critical", "High"]:

            alert_desc = f"Administrative Risk Indicator flagged: {w_id} with score {r_score:.1f}/100. Primary factors: {', '.join(json.loads(drivers_json)[:2]) if drivers_json else 'Multiple risk triggers'}. Requires official verification."

            alert_rec = Alert(

                dataset_id=dataset_id,

                work_id=w_id,

                alert_type="Audit Risk Indicator",

                severity=r_tier,

                status="Open",

                description=alert_desc,

                assigned_to="District Nodal Officer"

            )

            db.session.add(alert_rec)



    # 7. Insert Duplicates

    if dup_pairs:

        for p in dup_pairs:

            db.session.add(DuplicateResult(

                dataset_id=dataset_id,

                work_id_1=str(p.get("work_id_1", "")),

                work_id_2=str(p.get("work_id_2", "")),

                similarity_score=float(p.get("similarity", 0.85)),

                matching_criteria=f"TF-IDF textual & regional similarity ({p.get('similarity', 0.85)*100:.1f}%)"

            ))



    # Audit log

    db.session.add(AuditLog(

        username=uploaded_by,

        action="DATASET_INGESTION",

        details=f"Dataset {dataset_id} ('{name}') with {len(scored_df)} projects stored in SQL Server MPLADS_AI_DB."

    ))



    db.session.commit()

    clear_dataset_cache(dataset_id)

    return dataset, pipeline_res