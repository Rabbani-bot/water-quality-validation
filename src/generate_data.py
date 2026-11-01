"""River monitoring dataset (seeded) with known, injected data-entry errors.

All data are generated with a fixed seed. A clean "truth" table is generated first; errors are
then injected into a copy and every injected error is logged, so validation performance
can be measured exactly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 2024
STATIONS = pd.DataFrame({
    "station": [f"RS-{i:02d}" for i in range(11)],
    "river_km": np.arange(0, 22, 2),
})
OUTFALLS_KM = [8.0, 14.0]  # modelled industrial discharge points
PARAMS = ["pH", "DO_mgL", "BOD_mgL", "COD_mgL", "TSS_mgL", "temp_C"]


def _plume(km: np.ndarray) -> np.ndarray:
    """Pollution load downstream of outfalls, decaying with distance."""
    load = np.zeros_like(km, dtype=float)
    for o, strength in zip(OUTFALLS_KM, [1.0, 1.4]):
        d = km - o
        load += np.where(d >= 0, strength * np.exp(-d / 3.0), 0)
    return load


def generate_truth(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", "2024-12-30", freq="W-MON")
    rows = []
    for _, st in STATIONS.iterrows():
        p = _plume(np.array([st.river_km]))[0]
        for d in dates:
            season = np.sin(2 * np.pi * (d.month - 4) / 12)
            temp = 18 + 8 * season + rng.normal(0, 1.2)
            bod = max(1.0, rng.lognormal(np.log(6 + 90 * p), 0.25))
            cod = bod * rng.uniform(1.9, 2.6)
            tss = max(5.0, rng.lognormal(np.log(40 + 230 * p), 0.3))
            do = float(np.clip(9.5 - 0.15 * (temp - 15) - 3.5 * p + rng.normal(0, 0.4), 0.5, 14))
            ph = float(np.clip(7.6 - 0.6 * p + rng.normal(0, 0.15), 5.5, 9.5))
            storm = rng.random() < 0.02  # genuine extreme event: real data, NOT an error
            if storm:
                tss *= rng.uniform(4, 8); bod *= rng.uniform(1.8, 3); cod = bod * rng.uniform(1.9, 2.6)
            rows.append((st.station, st.river_km, d.date(), ph, do, bod, cod, tss, temp, storm))
    df = pd.DataFrame(rows, columns=["station", "river_km", "date", *PARAMS, "storm_event"])
    df["record_id"] = np.arange(len(df))
    return df


def inject_errors(truth: pd.DataFrame, rate: float = 0.15, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject errors into `rate` share of records. Returns (dirty, error_log)."""
    rng = np.random.default_rng(seed + 1)
    dirty = truth.copy()
    dirty["date"] = dirty["date"].astype(str)
    n = len(dirty)
    idx = rng.choice(n, size=int(rate * n), replace=False)
    types = ["unit_ug_for_mg", "decimal_shift", "impossible_value", "station_typo",
             "missing_value", "future_date", "bod_gt_cod", "subtle_transcription"]
    log = []
    for i in idx:
        t = types[rng.integers(len(types))]
        if t == "unit_ug_for_mg":
            col = rng.choice(["BOD_mgL", "COD_mgL", "TSS_mgL"]); dirty.loc[i, col] *= 1000
        elif t == "decimal_shift":
            col = rng.choice(["DO_mgL", "BOD_mgL", "TSS_mgL"]); dirty.loc[i, col] *= 10
        elif t == "impossible_value":
            col = rng.choice(["pH", "DO_mgL", "temp_C"])
            dirty.loc[i, col] = {"pH": 17.2, "DO_mgL": -3.0, "temp_C": 65.0}[col]
        elif t == "station_typo":
            col = "station"; dirty.loc[i, col] = dirty.loc[i, "station"].replace("RS-", "RS")
        elif t == "missing_value":
            col = rng.choice(PARAMS); dirty.loc[i, col] = np.nan
        elif t == "future_date":
            col = "date"; dirty.loc[i, col] = "2042-" + dirty.loc[i, "date"][5:]
        elif t == "bod_gt_cod":
            col = "COD_mgL"; dirty.loc[i, col] = dirty.loc[i, "BOD_mgL"] * 0.5
        elif t == "subtle_transcription":
            col = rng.choice(["BOD_mgL", "TSS_mgL", "DO_mgL"]); dirty.loc[i, col] *= rng.choice([0.7, 1.3])
        log.append((int(dirty.loc[i, "record_id"]), t, col))
    # exact duplicate submissions (extra rows, also errors)
    dup_src = rng.choice(np.setdiff1d(np.arange(n), idx), size=int(0.02 * n), replace=False)
    dups = dirty.iloc[dup_src].copy()
    for rid in dups["record_id"]:
        log.append((int(rid), "duplicate", "row"))
    dirty = pd.concat([dirty, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return dirty, pd.DataFrame(log, columns=["record_id", "error_type", "column"])


if __name__ == "__main__":
    t = generate_truth()
    d, log = inject_errors(t)
    t.to_csv(ROOT / "data" / "truth.csv", index=False)
    d.to_csv(ROOT / "data" / "monitoring_raw.csv", index=False)
    log.to_csv(ROOT / "data" / "injected_errors.csv", index=False)
    STATIONS.to_csv(ROOT / "data" / "stations.csv", index=False)
    print(len(t), len(d), len(log))
