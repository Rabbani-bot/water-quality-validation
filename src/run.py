"""End-to-end run: generate data, validate, evaluate, map exceedance hotspots."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_truth, inject_errors, STATIONS, OUTFALLS_KM  # noqa: E402
from validate import validate, evaluate, exceedances  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> dict:
    truth = generate_truth()
    raw, log = inject_errors(truth)
    truth.to_csv(ROOT / "data" / "truth.csv", index=False)
    raw.to_csv(ROOT / "data" / "monitoring_raw.csv", index=False)
    log.to_csv(ROOT / "data" / "injected_errors.csv", index=False)
    STATIONS.to_csv(ROOT / "data" / "stations.csv", index=False)

    v = validate(raw, set(STATIONS["station"]), ("2022-01-01", "2024-12-31"))
    metrics = evaluate(v, log)
    rule_cols = ["missing", "unknown_station", "date_out_of_window", "physical_range",
                 "bod_gt_cod", "duplicate", "statistical_outlier"]
    metrics["flags_by_rule"] = {c: int(v[c].sum()) for c in rule_cols}

    # recall by injected error type
    by_type = {}
    flagged_ids = set(v.loc[v["flagged"], "record_id"])
    for t, g in log[log["error_type"] != "duplicate"].groupby("error_type"):
        by_type[t] = round(100 * g["record_id"].isin(flagged_ids).mean(), 1)
    dup_rows = v[v.duplicated(subset=["record_id"], keep="first")]
    by_type["duplicate"] = round(100 * dup_rows["flagged"].mean(), 1) if len(dup_rows) else None
    metrics["recall_by_error_type_pct"] = by_type

    clean = v[~v["flagged"]].drop(columns=rule_cols + ["flagged", "date_parsed"])
    clean.to_csv(ROOT / "data" / "monitoring_clean.csv", index=False)
    v.to_csv(OUT / "validation_flags.csv", index=False)
    ex = exceedances(clean)
    ex.to_csv(OUT / "exceedance_by_station.csv", index=False)
    worst = ex.sort_values("BOD_mgL_exceed_pct", ascending=False).iloc[0]
    metrics["worst_station"] = {"station": worst["station"], "river_km": float(worst["river_km"]),
                                "BOD_exceed_pct": float(worst["BOD_mgL_exceed_pct"]),
                                "BOD_median_excess_pct": float(worst["BOD_mgL_median_excess_pct"])}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("River water-quality validation and hotspot screening",
                 fontweight="bold")
    a = ax[0]
    counts = pd.Series(metrics["flags_by_rule"]).sort_values()
    a.barh(counts.index, counts.values, color="#2E6F95")
    a.set_title(f"Records flagged by rule\nprecision {metrics['precision_pct']}%, recall {metrics['recall_pct']}%")
    a = ax[1]
    a.bar(["before", "after validation"], [metrics["error_rate_before_pct"], metrics["residual_error_rate_pct"]],
          color=["#BC4749", "#6A994E"])
    for i, val in enumerate([metrics["error_rate_before_pct"], metrics["residual_error_rate_pct"]]):
        a.text(i, val, f"{val:.2f}%", ha="center", va="bottom")
    a.set_ylabel("% of records containing an error"); a.set_title("Error rate (measured against injected-error log)")
    a = ax[2]
    a.plot(ex["river_km"], ex["BOD_mgL_exceed_pct"], marker="o", label="BOD > 80 mg/L")
    a.plot(ex["river_km"], ex["COD_mgL_exceed_pct"], marker="s", label="COD > 150 mg/L")
    a.plot(ex["river_km"], ex["TSS_mgL_exceed_pct"], marker="^", label="TSS > 200 mg/L")
    for o in OUTFALLS_KM:
        a.axvline(o, color="grey", ls="--"); a.text(o, 102, "outfall", ha="center", fontsize=8)
    a.set_ylim(0, 110); a.set_xlabel("river km (downstream)"); a.set_ylabel("% of samples exceeding")
    a.set_title("Exceedance hotspots along a 20 km stretch"); a.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "dashboard.png", dpi=130); plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
