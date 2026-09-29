"""Duplicate Work Detection Engine
Detects redundant, ghost, or duplicated municipal and rural civil works
using TF-IDF n-gram vectorization and Cosine Similarity partitioned
by administrative district and project category.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_duplicate_works(df: pd.DataFrame, threshold: float = 0.80) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Identifies candidate duplicate works with lexical cosine similarity >= threshold.
    
    Returns:
        (df_with_duplicate_flag, duplicate_pairs_list)
    """
    dup_df = df.copy()
    dup_df["duplicate_risk_flag"] = 0
    duplicate_pairs = []
    flagged_work_ids = set()

    if len(dup_df) == 0 or "work_description" not in dup_df.columns:
        return dup_df, duplicate_pairs

    # Group by (district, work_category) to localize duplicate comparisons
    group_cols = ["district", "work_category"]
    has_groups = all(c in dup_df.columns for c in group_cols)
    
    if has_groups:
        groups = dup_df.groupby(group_cols)
    else:
        # Fallback to single group if columns absent
        groups = [("All", dup_df)]

    for group_key, grp in groups:
        if len(grp) < 2:
            continue
            
        descriptions = grp["work_description"].fillna("").astype(str).tolist()
        work_ids = grp["work_id"].tolist()
        sanctioned_amts = grp.get("sanctioned_amount", pd.Series([0.0]*len(grp))).tolist()
        expenditures = grp.get("expenditure", pd.Series([0.0]*len(grp))).tolist()
        agencies = grp.get("implementing_agency", pd.Series([""]*len(grp))).tolist()
        
        # Check non-empty descriptions
        valid_indices = [i for i, d in enumerate(descriptions) if len(d.strip()) > 5]
        if len(valid_indices) < 2:
            continue
            
        sub_desc = [descriptions[i] for i in valid_indices]
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b", min_df=1)
            tfidf_mat = vectorizer.fit_transform(sub_desc)
            sim_mat = cosine_similarity(tfidf_mat)
        except Exception:
            continue

        n_sub = len(valid_indices)
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                sim = sim_mat[i, j]
                if sim >= threshold:
                    idx_i = valid_indices[i]
                    idx_j = valid_indices[j]
                    
                    id_1 = str(work_ids[idx_i])
                    id_2 = str(work_ids[idx_j])
                    
                    flagged_work_ids.add(id_1)
                    flagged_work_ids.add(id_2)
                    
                    dist_val = grp["district"].iloc[0] if "district" in grp.columns else "Unknown"
                    cat_val = grp["work_category"].iloc[0] if "work_category" in grp.columns else "Unknown"
                    
                    duplicate_pairs.append({
                        "work_id_1": id_1,
                        "work_description_1": descriptions[idx_i],
                        "amount_1": sanctioned_amts[idx_i],
                        "expenditure_1": expenditures[idx_i],
                        "agency_1": agencies[idx_i],
                        "work_id_2": id_2,
                        "work_description_2": descriptions[idx_j],
                        "amount_2": sanctioned_amts[idx_j],
                        "expenditure_2": expenditures[idx_j],
                        "agency_2": agencies[idx_j],
                        "district": dist_val,
                        "work_category": cat_val,
                        "similarity_score": round(float(sim), 4),
                        "similarity_pct": round(float(sim) * 100, 1)
                    })

    # Sort pairs by similarity descending
    duplicate_pairs.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Set binary duplicate risk flag on dataframe
    dup_df["duplicate_risk_flag"] = dup_df["work_id"].isin(flagged_work_ids).astype(int)

    return dup_df, duplicate_pairs
