"""Rule-based + statistical validation for water-quality monitoring records."""
import numpy as np
import pandas as pd

PARAMS = ["pH", "DO_mgL", "BOD_mgL", "COD_mgL", "TSS_mgL", "temp_C"]
# Physically plausible ranges for river water (hard limits)
PHYSICAL = {"pH": (0, 14), "DO_mgL": (0, 20), "BOD_mgL": (0, 1000), "COD_mgL": (0, 2000),
            "TSS_mgL": (0, 5000), "temp_C": (0, 40)}
# Screening thresholds. pH, BOD, COD, TSS: Pakistan NEQS limits for effluent discharge to inland
# waters, used here only as screening values; DO: common minimum for aquatic life (representative).
THRESHOLDS = {"pH": (6.0, 9.0), "BOD_mgL": (None, 80), "COD_mgL": (None, 150),
              "TSS_mgL": (None, 200), "DO_mgL": (5.0, None)}


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


def validate(df: pd.DataFrame, stations: set[str], period: tuple[str, str], z_limit: float = 5.0) -> pd.DataFrame:
    """Return one row per record with boolean rule flags and an overall `flagged` column."""
    v = df.drop(columns=["storm_event"], errors="ignore").copy()
    v["date_parsed"] = pd.to_datetime(v["date"], errors="coerce")
    f = pd.DataFrame(index=v.index)
    f["missing"] = v[PARAMS].isna().any(axis=1)
    f["unknown_station"] = ~v["station"].isin(stations)
    f["date_out_of_window"] = ~v["date_parsed"].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))
    f["physical_range"] = False
    for c, (lo, hi) in PHYSICAL.items():
        f["physical_range"] |= v[c].notna() & ((v[c] < lo) | (v[c] > hi))
    f["bod_gt_cod"] = v["BOD_mgL"] > v["COD_mgL"]
    f["duplicate"] = v.duplicated(subset=["station", "date", *PARAMS], keep="first")
    # statistical outliers within each station (log scale for skewed concentrations)
    f["statistical_outlier"] = False
    for c in PARAMS:
        x = np.log10(v[c].clip(lower=1e-3)) if c in ("BOD_mgL", "COD_mgL", "TSS_mgL") else v[c]
        z = x.groupby(v["station"]).transform(robust_z)
        f["statistical_outlier"] |= z.abs() > z_limit
    f["flagged"] = f.any(axis=1)
    return pd.concat([v, f], axis=1)


def evaluate(validated: pd.DataFrame, error_log: pd.DataFrame) -> dict:
    """Compare flags with the injected-error log. Duplicates: only the extra copy is an error."""
    err_ids = set(error_log.loc[error_log["error_type"] != "duplicate", "record_id"])
    v = validated.copy()
    first = ~v.duplicated(subset=["record_id"], keep="first")
    v["is_error"] = (v["record_id"].isin(err_ids) & first) | ~first
    tp = int((v["flagged"] & v["is_error"]).sum())
    fp = int((v["flagged"] & ~v["is_error"]).sum())
    fn = int((~v["flagged"] & v["is_error"]).sum())
    before = v["is_error"].mean()
    after = v.loc[~v["flagged"], "is_error"].mean()
    return {
        "records": len(v), "true_errors": int(v["is_error"].sum()),
        "error_rate_before_pct": round(100 * before, 2),
        "flagged": int(v["flagged"].sum()),
        "precision_pct": round(100 * tp / (tp + fp), 1) if tp + fp else None,
        "recall_pct": round(100 * tp / (tp + fn), 1) if tp + fn else None,
        "residual_error_rate_pct": round(100 * after, 2),
        "false_positives": fp, "missed_errors": fn,
    }


def exceedances(clean: pd.DataFrame) -> pd.DataFrame:
    """Per-station exceedance frequency and median % above the screening limit."""
    out = []
    for st, g in clean.groupby("station"):
        row = {"station": st, "river_km": g["river_km"].iloc[0], "n": len(g)}
        for c, (lo, hi) in THRESHOLDS.items():
            if hi is not None:
                ex = g[c] > hi
                row[f"{c}_exceed_pct"] = round(100 * ex.mean(), 1)
                row[f"{c}_median_excess_pct"] = round(100 * ((g.loc[ex, c] / hi) - 1).median(), 1) if ex.any() else 0.0
            else:
                row[f"{c}_below_min_pct"] = round(100 * (g[c] < lo).mean(), 1)
        out.append(row)
    return pd.DataFrame(out).sort_values("river_km")
