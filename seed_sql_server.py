"""Seed and Populate Microsoft SQL Server (MPLADS_AI_DB)

Initializes database schema, seeds the authorized admin account and standard user account,

seeds default system settings, and populates initial audited datasets:

1. DS-2024-NAT-001 (National MPLADS Comprehensive Audit FY 2024-25)

2. DS-2023-PILOT-002 (State Pilot Priority Infrastructure Audit FY 2023-24)

"""

import os

import sys

import json

from pathlib import Path

from datetime import datetime



BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:

    sys.path.insert(0, str(BASE_DIR))



import pandas as pd

import numpy as np



from config import Config

from app import create_app

from app.models_db import (

    db, User, Dataset, MP, District, Agency, Project,

    FinancialRecord, ImplementationRecord, DataUpload,

    DataValidationResult, DataQualityResult, AnomalyResult,

    DuplicateResult, RiskScore, Alert, SystemSetting, AuditLog

)

from src.load_data import load_dataset

from src.pipeline import run_pipeline



def seed_system_settings():

    """Seeds default weights, risk thresholds, and Power BI config in SQL Server."""

    defaults = [

        ("risk_weights", json.dumps({

            "if": Config.WEIGHT_ISOLATION_FOREST,

            "svm": Config.WEIGHT_ONE_CLASS_SVM,

            "rf": Config.WEIGHT_RANDOM_FOREST,

            "rules": Config.WEIGHT_RULES

        })),

        ("risk_thresholds", json.dumps(Config.RISK_THRESHOLDS)),

        ("powerbi_url", Config.POWERBI_URL),

        ("disclaimer_notice", "NOTICE: The flags and scores displayed are algorithmic risk indicators calculated for administrative prioritization and official verification. They do NOT constitute legal determination or automatic proof of fraud.")

    ]

    for key, val in defaults:

        existing = SystemSetting.query.filter_by(setting_key=key).first()

        if not existing:

            setting = SystemSetting(setting_key=key, setting_value=val, updated_by="System")

            db.session.add(setting)

    db.session.commit()

    print("System settings verified in SQL Server.")





def seed_users():

    """Seeds authorized accounts using environment-provided passwords."""

    admin = User.query.filter_by(username="admin").first()



    if not admin:

        admin_password = os.environ.get("MPLADS_ADMIN_PASSWORD")

        if not admin_password:

            raise RuntimeError(

                "Set MPLADS_ADMIN_PASSWORD before seeding the admin account."

            )



        admin = User(

            username="admin",

            email="admin@example.com",

            full_name="Project Administrator",

            role="admin"

        )

        admin.set_password(admin_password)

        db.session.add(admin)

        print("Admin account created.")



    else:

        admin.role = "admin"



    user = User.query.filter_by(username="user").first()



    if not user:

        user_password = os.environ.get("MPLADS_USER_PASSWORD")

        if not user_password:

            raise RuntimeError(

                "Set MPLADS_USER_PASSWORD before seeding the user account."

            )



        user = User(

            username="user",

            email="user@example.com",

            full_name="Audit Monitoring Officer",

            role="user"

        )

        user.set_password(user_password)

        db.session.add(user)

        print("Standard user account created.")



    else:

        user.role = "user"



    db.session.commit()





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



def ingest_dataset_to_sql_server(dataset_id: str, name: str, description: str, csv_path: Path, uploaded_by: str = "admin"):

    """

    Runs full AI validation & analysis pipeline on a CSV dataset and persists all

    relational entities into SQL Server partitioned by dataset_id.

    """

    print(f"\nProcessing dataset '{name}' ({dataset_id}) from {csv_path.name}...")



    # Check if dataset already populated

    existing_ds = Dataset.query.filter_by(dataset_id=dataset_id).first()

    if existing_ds:

        print(f"Cleaning existing records for {dataset_id}...")

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

    raw_df, meta = load_dataset(csv_path)



    # Run full AI pipeline

    pipeline_res = run_pipeline(raw_df, data_source="Audited Portal", force_retrain=True)

    scored_df = pipeline_res["scored_df"]

    val_rep = pipeline_res["validation_report"]

    qual_rep = pipeline_res["quality_report"]

    dup_pairs = pipeline_res["duplicate_pairs"]



    # 1. Insert Dataset Record

    dataset = Dataset(

        dataset_id=dataset_id,

        name=name,

        description=description,

        record_count=len(scored_df),

        uploaded_by=uploaded_by,

        created_at=datetime.utcnow(),

        is_active=True,

        file_path=str(csv_path),

        data_source=meta.get("file_type", "CSV").upper()

    )

    db.session.add(dataset)

    db.session.flush()



    # 2. Insert DataUpload Record

    upload_rec = DataUpload(

        dataset_id=dataset_id,

        filename=csv_path.name,

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



    # 5. Insert MPs, Districts, Agencies

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



    # Commit dimensions

    db.session.flush()



    # 6. Insert Projects, FinancialRecords, ImplementationRecords, RiskScores, AnomalyResults, Alerts

    print(f"Inserting {len(scored_df)} project records and analysis entities into SQL Server...")



    for idx, row in scored_df.iterrows():

        w_id = str(row["work_id"])



        # Parse Dates safely

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



        # Project entity

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



        # Financial Record

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



        # Implementation Record

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



        # Risk Score Record

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



        # Anomaly Result Record

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



        # Alerts for High and Critical tiers

        if r_tier in ["Critical", "High"]:

            alert_desc = f"Administrative Risk Indicator flagged: {w_id} with score {r_score:.1f}/100. Primary factors: {', '.join(json.loads(drivers_json)[:2]) if drivers_json else 'Multiple risk triggers'}. Requires field verification."

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

        print(f"Inserting {len(dup_pairs)} duplicate work detection pairs into SQL Server...")

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

        details=f"Dataset {dataset_id} ('{name}') with {len(scored_df)} records successfully analyzed and stored in SQL Server."

    ))



    db.session.commit()

    print(f"Dataset '{name}' successfully ingested and committed to SQL Server MPLADS_AI_DB!")

    return dataset



def main():

    app = create_app()

    with app.app_context():

        print("Initializing tables in Microsoft SQL Server MPLADS_AI_DB...")

        db.create_all()

        print("Tables verified/created successfully.")



        seed_system_settings()

        seed_users()



        # Ingest Dataset 1: National Comprehensive Audit

        ds1_path = BASE_DIR / "data" / "synthetic" / "mplads_synthetic_500.csv"

        if ds1_path.exists():

            ingest_dataset_to_sql_server(

                dataset_id="DS-2024-NAT-001",

                name="National MPLADS Comprehensive Audit FY 2024-25",

                description="Comprehensive multi-state audit covering 500 infrastructure projects across roads, community assets, healthcare, and education with complete financial and physical metrics.",

                csv_path=ds1_path

            )



        # Ingest Dataset 2: State Pilot Priority Infrastructure

        ds2_path = BASE_DIR / "data" / "synthetic" / "mplads_synthetic_200.csv"

        if ds2_path.exists():

            ingest_dataset_to_sql_server(

                dataset_id="DS-2023-PILOT-002",

                name="State Pilot Priority Infrastructure Audit FY 2023-24",

                description="Focused state-level priority audit inspecting rural connectivity, educational infrastructure, and municipal drainage works across key developmental blocks.",

                csv_path=ds2_path

            )



        print("\nAll database tables and seed datasets verified and stored in SQL Server MPLADS_AI_DB!")



if __name__ == '__main__':

    main()