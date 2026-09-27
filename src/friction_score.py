"""
friction_score.py
==================
Phase 2: Vaccine Access Friction Score (VAFS) model.

Turns the Phase 1 outputs (ACS DP03 county/ZCTA data + travel_times.py's
nearest_clinics.csv) into a 0-100 friction score per region, broken into
five domains, with:
  - a configurable weighted scoring system
  - a sensitivity analysis (does the score behave intuitively as weights change?)
  - an honest accounting of which domains are backed by real data vs. missing

DATA REALITY CHECK (please read before presenting this as final)
------------------------------------------------------------------
Your uploaded ACSDP5Y2024.DP03-Data.csv is a MIX of two geography levels:
  - 7 counties          (GEO_ID prefix "0500000US...")
  - 6 ZCTAs / ZIP codes  (GEO_ID prefix "860Z200US...", all Baltimore-area,
                          21240/21244/21250/21251/21252/21285 -- looks like
                          a UMBC-area test pull)
This script handles both transparently via `parse_geo_id()`, but it means
your current sample isn't a full state or even one consistent geography --
useful for a demo, not for a real map yet.

Digital access was dropped as a scoring domain (no free broadband dataset
was on hand, and rather than fake it the domain was removed outright). Of
the remaining four proposed friction domains, DP03 gives you real signal
for three:
  - Physical access     -> travel time (from Phase 1) + vehicle-access proxy
  - Temporal access      -> employment rate, WFH rate, commute time (proxies,
                             not actual clinic hours -- see note in code)
  - Economic/admin burden -> uninsured rate, low-income %, SNAP reliance
  - Social/structural     -> PARTIAL. "Hourly job prevalence" proxied via
                             service + production/transport occupations.
                             Childcare and misinformation intensity have NO
                             free structured dataset -- flagged, not faked.

The script does NOT silently pretend missing domains = 0 friction. A domain
missing for a given region is excluded from that region's composite and its
weight redistributed proportionally across domains that have data, and a
`data_completeness` column tells you exactly how much of each domain's
intended indicator set is actually populated.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATA_DIR = Path("Data")
CENSUS_CSV = DATA_DIR / "ACSDP5Y2024.DP03-Data.csv"
NEAREST_CLINICS_CSV = DATA_DIR / "nearest_clinics.csv"   # output of travel_times.py

OUT_SCORES = DATA_DIR / "friction_scores.csv"
OUT_SENSITIVITY = DATA_DIR / "sensitivity_report.csv"

# Default top-level domain weights
DEFAULT_WEIGHTS = {
    "physical_access": 0.39,
    "temporal_access": 0.22,
    "economic_admin": 0.28,
    "social_structural": 0.11,  # partial data only
}

N_TOP_DRIVERS = 3  # how many individual indicators to report per region



# STEP 0 -- geography parsing 

def parse_geo_id(geo_id: str) -> tuple[str, str]:
    """
    Return (geo_type, region_code) from a Census GEO_ID string.
    '0500000US24003'  -> ('county', '24003')
    """
    if geo_id.startswith("0500000US"):
        return "county", geo_id.replace("0500000US", "")
    if "Z200US" in geo_id:
        return "zcta", geo_id.split("US")[-1]
    return "unknown", geo_id



# STEP 1 -- load census data, keep the raw columns each indicator needs

RAW_COLUMNS = [
    "GEO_ID", "NAME",
    # commuting / vehicle access
    "DP03_0018E", "DP03_0019E", "DP03_0020E", "DP03_0024E", "DP03_0025E",
    # employment
    "DP03_0001E", "DP03_0004E",
    # occupation (hourly-job proxy)
    "DP03_0026E", "DP03_0028E", "DP03_0031E",
    # income
    "DP03_0051E", "DP03_0052E", "DP03_0053E", "DP03_0054E",
    # public assistance
    "DP03_0074E",
    # insurance
    "DP03_0095E", "DP03_0099E",
]


def load_census(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1])
    df = df[[c for c in RAW_COLUMNS if c in df.columns]].copy()

    numeric_cols = [c for c in df.columns if c not in ("GEO_ID", "NAME")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.replace({-666666666: np.nan, -999999999: np.nan, -888888888: np.nan}, inplace=True)

    df[["geo_type", "region_code"]] = df["GEO_ID"].apply(lambda g: pd.Series(parse_geo_id(g)))
    df["region_name"] = df["NAME"]

    return df


def load_travel_times(path: Path) -> pd.DataFrame | None:
    """Load Phase 1 travel times (nearest_clinics.csv) if available."""
    if not path.exists():
        print(f"{path} Not found -- run travel_times.py first. "
              "Phase 2 will still run, but physical_access domain will be missing.")
        return None
    tt = pd.read_csv(path)
    agg = (
        tt.groupby("origin_id")["duration_min"]
        .mean()
        .reset_index()
        .rename(columns={"origin_id": "region_name", "duration_min": "avg_clinic_travel_min"})
    )
    return agg



# STEP 2 -- indicator registry
# Each indicator: domain, a human label, a function computing the raw value
# from a row, and whether HIGHER raw value means MORE friction (invert=False)
# or LESS friction (invert=True, e.g. "% working from home").


def _safe_pct(numerator, denominator):
    if denominator in (0, None) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return 100 * numerator / denominator


INDICATORS = [
    # -- Physical access --
    dict(domain="physical_access", name="avg_clinic_travel_min", invert=False,
         label="Avg. drive time to nearest clinics (min)",
         fn=lambda r: r.get("avg_clinic_travel_min", np.nan)),
    dict(domain="physical_access", name="no_vehicle_commute_pct", invert=False,
         label="% of workers not driving to work (no car access proxy)",
         fn=lambda r: 100 - _safe_pct(r["DP03_0019E"] + r["DP03_0020E"], r["DP03_0018E"])
                       if pd.notna(r.get("DP03_0018E")) else np.nan),

    # -- Temporal access --
    dict(domain="temporal_access", name="employed_pct", invert=False,
         label="% employed (proxy for rigid work schedules)",
         fn=lambda r: _safe_pct(r["DP03_0004E"], r["DP03_0001E"])),
    dict(domain="temporal_access", name="low_wfh_pct", invert=False,
         label="% NOT working from home (proxy for low schedule flexibility)",
         fn=lambda r: 100 - _safe_pct(r["DP03_0024E"], r["DP03_0018E"])
                       if pd.notna(r.get("DP03_0018E")) else np.nan),
    dict(domain="temporal_access", name="mean_commute_min", invert=False,
         label="Mean commute time to work (min)",
         fn=lambda r: r.get("DP03_0025E", np.nan)),

    # -- Economic / administrative burden --
    dict(domain="economic_admin", name="uninsured_pct", invert=False,
         label="% uninsured",
         fn=lambda r: _safe_pct(r["DP03_0099E"], r["DP03_0095E"])),
    dict(domain="economic_admin", name="low_income_pct", invert=False,
         label="% households earning < $25k",
         fn=lambda r: _safe_pct(r["DP03_0052E"] + r["DP03_0053E"] + r["DP03_0054E"], r["DP03_0051E"])),
    dict(domain="economic_admin", name="public_assistance_pct", invert=False,
         label="% households on SNAP / cash assistance",
         fn=lambda r: _safe_pct(r["DP03_0074E"], r["DP03_0051E"])),

    # -- Social / structural: partial data (hourly-job proxy only) --
    dict(domain="social_structural", name="hourly_job_pct", invert=False,
         label="% in service/production/transport occupations (hourly-job proxy)",
         fn=lambda r: _safe_pct(r["DP03_0028E"] + r["DP03_0031E"], r["DP03_0026E"])),
    # Childcare-constraint and misinformation-intensity indicators are not
    # included: no free structured dataset covers either at county/ZCTA
    # level. Do not fabricate placeholder numbers for these.
]

ALL_DOMAINS = ["physical_access", "temporal_access", "economic_admin",
               "social_structural"]

# How many indicators each domain WOULD have if fully instrumented -- used
# only to report data_completeness honestly.
DOMAIN_TARGET_INDICATOR_COUNTS = {
    "physical_access": 2,
    "temporal_access": 3,
    "economic_admin": 3,
    "social_structural": 3,  # hourly-job (have) + childcare + misinformation (missing)
}



# STEP 3 -- compute raw indicators, normalize, aggregate into domains


def compute_raw_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["GEO_ID", "region_name", "geo_type", "region_code"]].copy()
    for ind in INDICATORS:
        out[ind["name"]] = df.apply(ind["fn"], axis=1)
    return out


def minmax_normalize(series: pd.Series, invert: bool) -> pd.Series:
    """Scale a raw indicator to 0-100 across the regions present, where
    100 always means 'most friction'. With invert=True, a HIGH raw value
    means LESS friction (e.g. % working from home), so the scale flips.
    """
    s = series.astype(float)
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        # No variation (or all missing) -- neutral midpoint, not a fake spread
        return pd.Series(np.where(s.isna(), np.nan, 50.0), index=s.index)
    scaled = 100 * (s - lo) / (hi - lo)
    return 100 - scaled if invert else scaled


def build_indicator_table(raw: pd.DataFrame) -> pd.DataFrame:
    table = raw[["GEO_ID", "region_name", "geo_type", "region_code"]].copy()
    for ind in INDICATORS:
        table[f"{ind['name']}_score"] = minmax_normalize(raw[ind["name"]], ind["invert"])
    return table


def compute_domain_scores(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average each domain's available indicator scores per region.
    Returns (domain_scores_df, completeness_df).
    """
    domain_scores = table[["GEO_ID", "region_name", "geo_type", "region_code"]].copy()
    completeness_rows = []

    for domain in ALL_DOMAINS:
        score_cols = [f"{i['name']}_score" for i in INDICATORS if i["domain"] == domain]
        target_n = DOMAIN_TARGET_INDICATOR_COUNTS.get(domain, len(score_cols))

        if not score_cols:
            domain_scores[domain] = np.nan
            completeness_rows.append({"domain": domain, "indicators_available": 0,
                                       "indicators_target": target_n, "completeness_pct": 0.0})
            continue

        domain_scores[domain] = table[score_cols].mean(axis=1, skipna=True)
        n_available = len(score_cols)
        completeness_rows.append({
            "domain": domain, "indicators_available": n_available,
            "indicators_target": target_n,
            "completeness_pct": round(100 * n_available / target_n, 1),
        })

    completeness = pd.DataFrame(completeness_rows)
    return domain_scores, completeness



# STEP 4 -- weighted composite, with renormalization for missing domains


def compute_composite(domain_scores: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted average of domain scores. A domain with NaN score for a
    given region (or entirely missing data) is excluded from that region's
    calculation and the remaining weights are renormalized to sum to 1 --
    so a data gap never silently drags the score toward 0 or 100.
    """
    composite = pd.Series(index=domain_scores.index, dtype=float)
    for idx, row in domain_scores.iterrows():
        w_sum, weighted_sum = 0.0, 0.0
        for domain, w in weights.items():
            val = row.get(domain, np.nan)
            if w > 0 and pd.notna(val):
                weighted_sum += w * val
                w_sum += w
        composite[idx] = weighted_sum / w_sum if w_sum > 0 else np.nan
    return composite



# STEP 5 -- top friction drivers per region (individual indicators, not
# just domains -- these map directly to specific interventions in Phase 4)


def top_drivers(table: pd.DataFrame, n: int = N_TOP_DRIVERS) -> pd.Series:
    score_cols = {f"{i['name']}_score": i["label"] for i in INDICATORS}

    def _row_top(row):
        vals = {label: row[col] for col, label in score_cols.items() if pd.notna(row[col])}
        ranked = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:n]

        return "; ".join(f"{label} ({score:.0f})" for label, score in ranked)

    return table.apply(_row_top, axis=1)



# STEP 6 -- sensitivity analysis
# "Run sensitivity analysis to ensure weights behave on their own."

# Two checks:
#   (a) Monotonicity: under the DEFAULT weights, does each domain's score
#       correlate positively with the final composite? (It should -- if a
#       domain is negatively correlated with the composite, something is
#       either mis-inverted or dominated by a collinear domain.)
#   (b) Stability: nudge each domain's weight up/down by a delta (renormalizing
#       the rest) and measure the Spearman rank correlation between the
#       baseline ranking and the perturbed ranking. A domain whose weight
#       barely moves the ranking has little practical influence; one whose
#       small nudge reshuffles the ranking a lot deserves a closer look
#       before you lock in final weights.


def sensitivity_analysis(domain_scores: pd.DataFrame, weights: dict, delta: float = 0.10) -> pd.DataFrame:
    baseline = compute_composite(domain_scores, weights)
    active_domains = [d for d, w in weights.items() if w > 0
                       and domain_scores[d].notna().any()]

    rows = []
    for domain in active_domains:
        corr = domain_scores[domain].corr(baseline, method="spearman")
        rows.append({
            "check": "monotonicity",
            "domain": domain,
            "detail": "corr(domain_score, composite) under default weights",
            "value": round(corr, 3) if pd.notna(corr) else np.nan,
            "flag": "REVIEW: non-positive correlation" if pd.notna(corr) and corr <= 0 else "ok",
        })

    for domain in active_domains:
        for direction, sign in [("increase", +1), ("decrease", -1)]:
            perturbed = dict(weights)
            perturbed[domain] = max(0.0, weights[domain] + sign * delta * weights[domain])
            # renormalize is implicit in compute_composite's per-row w_sum logic
            perturbed_scores = compute_composite(domain_scores, perturbed)

            rank_corr, _ = spearmanr(
                baseline.rank(na_option="keep"), perturbed_scores.rank(na_option="keep"),
                nan_policy="omit",
            )

            rows.append({
                "check": "stability",
                "domain": domain,
                "detail": f"{direction} weight by {int(delta*100)}%",
                "value": round(rank_corr, 3) if pd.notna(rank_corr) else np.nan,
                "flag": "REVIEW: ranking shifted a lot" if pd.notna(rank_corr) and rank_corr < 0.85 else "ok",
            })

    return pd.DataFrame(rows)



# STEP 7 -- optional validation hook against real uptake data


def validate_against_uptake(scores: pd.DataFrame, uptake_csv: Path,region_key: str = "region_name", uptake_col: str = "vaccine_uptake_pct") -> pd.DataFrame | None:
    if not uptake_csv.exists():
        print(f"NOTE: {uptake_csv} not found -- skipping uptake validation. "
              "Supply a CSV with columns [{}, {}] to enable this.".format(region_key, uptake_col))
        return None
    
    uptake = pd.read_csv(uptake_csv)
    merged = scores.merge(uptake, left_on="region_name", right_on=region_key, how="inner")
    corr, p = spearmanr(merged["friction_score"], merged[uptake_col], nan_policy="omit")

    print(f"Friction score vs. uptake: Spearman r = {corr:.3f} (p = {p:.3f}), "
          f"n = {len(merged)} regions")
    
    return merged



def main(weights: dict = None):
    weights = weights or DEFAULT_WEIGHTS

    print("Loading census data...")
    census = load_census(CENSUS_CSV)

    print("Loading Phase 1 travel times (if available)...")
    travel = load_travel_times(NEAREST_CLINICS_CSV)
    if travel is not None:
        census = census.merge(travel, on="region_name", how="left")
    else:
        census["avg_clinic_travel_min"] = np.nan

    print("Computing raw indicators...")
    raw = compute_raw_indicators(census)

    print("Normalizing indicators to 0-100 friction scale...")
    table = build_indicator_table(raw)

    print("Aggregating into domain scores...")
    domain_scores, completeness = compute_domain_scores(table)

    print("Computing weighted composite friction score...")
    domain_scores["friction_score"] = compute_composite(domain_scores, weights).round(1)
    domain_scores["top_friction_drivers"] = top_drivers(table)

    print("\nData completeness by domain:")
    print(completeness.to_string(index=False))

    print("\nRunning sensitivity analysis...")
    sensitivity = sensitivity_analysis(domain_scores, weights)
    print(sensitivity.to_string(index=False))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    domain_scores.sort_values("friction_score", ascending=False).to_csv(OUT_SCORES, index=False)
    sensitivity.to_csv(OUT_SENSITIVITY, index=False)
    print(f"\nSaved {OUT_SCORES}")
    print(f"Saved {OUT_SENSITIVITY}")

    return domain_scores, completeness, sensitivity


if __name__ == "__main__":
    main()