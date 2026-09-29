"""Data Loader & Schema Mapping Engine
Supports CSV, XLSX, XLS, and JSON formats.
Applies intelligent column matching and alias resolution to map diverse
government portal exports to the standardized canonical MPLADS schema.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from config import Config

def normalize_header(header: str) -> str:
    """Normalize string by lowercasing and removing non-alphanumeric chars."""
    if not isinstance(header, str):
        header = str(header)
    clean = re.sub(r"[^a-zA-Z0-9]", "_", header.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean

def map_columns_to_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Maps DataFrame columns to canonical column names using alias definitions.
    
    Returns:
        (mapped_df, mapping_dict, unmapped_columns)
    """
    mapped_df = df.copy()
    mapping_dict = {}
    unmapped_columns = []
    
    # Pre-compute normalized alias lookup
    # canonical_name -> list of normalized aliases
    norm_aliases = {}
    for canon, aliases in Config.COLUMN_ALIASES.items():
        norm_aliases[canon] = [normalize_header(a) for a in aliases]
        if normalize_header(canon) not in norm_aliases[canon]:
            norm_aliases[canon].append(normalize_header(canon))
            
    matched_canons = set()
    
    for original_col in df.columns:
        norm_col = normalize_header(original_col)
        matched_canon = None
        
        # Exact alias match
        for canon, aliases in norm_aliases.items():
            if canon not in matched_canons and norm_col in aliases:
                matched_canon = canon
                break
                
        # Partial substring match if not yet matched
        if not matched_canon:
            for canon, aliases in norm_aliases.items():
                if canon not in matched_canons:
                    for alias in aliases:
                        if len(alias) >= 4 and (alias in norm_col or norm_col in alias):
                            matched_canon = canon
                            break
                    if matched_canon:
                        break
                        
        if matched_canon:
            mapping_dict[original_col] = matched_canon
            matched_canons.add(matched_canon)
        else:
            unmapped_columns.append(original_col)
            
    # Rename mapped columns
    mapped_df = mapped_df.rename(columns=mapping_dict)
    
    # Fill missing canonical columns with None / empty string
    for canon in Config.CANONICAL_COLUMNS:
        if canon not in mapped_df.columns:
            mapped_df[canon] = None
            
    return mapped_df, mapping_dict, unmapped_columns

def detect_data_source(file_path: Union[str, Path], df: pd.DataFrame) -> str:
    """Identifies whether dataset is Synthetic or Uploaded."""
    path_str = str(file_path).lower()
    if "synthetic" in path_str or ("is_anomaly" in df.columns and "anomaly_type" in df.columns):
        return "Synthetic"
    return "Uploaded"

def load_dataset(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads dataset from CSV, XLSX, XLS, or JSON.
    Returns:
        (canonical_df, metadata_dict)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = path.suffix.lower()
    
    if ext == ".csv":
        # Handle potential encoding issues (UTF-8, ISO-8859-1, cp1252)
        try:
            raw_df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            raw_df = pd.read_csv(path, encoding="latin1")
    elif ext in [".xlsx", ".xls"]:
        raw_df = pd.read_excel(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            raw_df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if dict of lists or nested under 'data' or 'records'
            if "data" in data and isinstance(data["data"], list):
                raw_df = pd.DataFrame(data["data"])
            elif "records" in data and isinstance(data["records"], list):
                raw_df = pd.DataFrame(data["records"])
            else:
                raw_df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Allowed: {Config.ALLOWED_EXTENSIONS}")

    data_source = detect_data_source(path, raw_df)
    
    # Check if this dataset has anomaly labels
    has_labels = ("is_anomaly" in raw_df.columns)
    
    # Map to canonical schema
    mapped_df, mapping_used, unmapped = map_columns_to_canonical(raw_df)
    
    # Preserve anomaly labels if present
    if has_labels and "is_anomaly" not in mapped_df.columns:
        mapped_df["is_anomaly"] = raw_df["is_anomaly"]
    if "anomaly_type" in raw_df.columns and "anomaly_type" not in mapped_df.columns:
        mapped_df["anomaly_type"] = raw_df["anomaly_type"]
        
    metadata = {
        "file_name": path.name,
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "total_rows": len(mapped_df),
        "total_cols": len(mapped_df.columns),
        "data_source": data_source,
        "has_ground_truth": has_labels,
        "mapping_used": mapping_used,
        "unmapped_columns": unmapped
    }
    
    return mapped_df, metadata
