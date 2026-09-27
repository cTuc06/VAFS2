"""USAGE
-----
    python main.py                 run whatever hasn't been run yet
    python main.py --force         rerun every step from scratch
    python main.py --from-scores   skip straight to building the map
                                    (use this if Data/friction_scores.csv
                                    already exists and you just want to
                                    regenerate index.html)
"""
 
from __future__ import annotations
 
import argparse
import os
import shutil
import sys
from pathlib import Path
 
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
 
sys.path.insert(0, str(ROOT))
 
 
def _have(*names: str) -> bool:
    return all((DATA_DIR / n).exists() for n in names)
 
 
def _ensure_census_csv() -> None:
    """The scripts expect Data/ACSDP5Y2024.DP03-Data.csv (dot). If only the
    underscore-named upload is present, copy it to the expected name rather
    than editing three scripts' hardcoded paths."""
    wanted = DATA_DIR / "ACSDP5Y2024.DP03-Data.csv"
    alt = DATA_DIR / "ACSDP5Y2024_DP03-Data.csv"
    if not wanted.exists() and alt.exists():
        shutil.copyfile(alt, wanted)
        print(f"[setup] copied {alt.name} -> {wanted.name}")
 
 
def step_clinics(force: bool) -> None:
    out = "md_vaccination_site_distances.csv"
    if not force and _have(out):
        print(f"[skip] Data/{out} already exists")
        return
    print("\n=== Step 1/4: geocoding vaccination sites (clinic_Locs.py) ===")
    # clinic_Locs.py runs its work at import time (no main() guard),
    # so importing it is how it executes.
    import clinic_Locs  # noqa: F401
 
 
def step_travel_times(force: bool) -> None:
    if not force and _have("travel_times.csv", "nearest_clinics.csv"):
        print("[skip] Data/travel_times.csv + nearest_clinics.csv already exist")
        return
    print("\n=== Step 2/4: building travel-time matrix (travel_times.py) ===")
    import travel_times
    travel_times.main()
 
 
def step_friction_score(force: bool) -> None:
    if not force and _have("friction_scores.csv", "sensitivity_report.csv"):
        print("[skip] Data/friction_scores.csv + sensitivity_report.csv already exist")
        return
    print("\n=== Step 3/4: computing friction scores (friction_score.py) ===")
    import friction_score
    friction_score.main()
 
 
def step_build_map(force: bool) -> Path:
    out = DATA_DIR / "vafs_map.html"
    if not force and out.exists():
        print("[skip] Data/vafs_map.html already exists")
    else:
        print("\n=== Step 4/4: building the map (build_vafs_map.py) ===")
        import build_vafs_map
        build_vafs_map.main()
    return out
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VAFS pipeline end to end.")
    parser.add_argument("--force", action="store_true",
                         help="rerun every step, ignoring existing output files")
    parser.add_argument("--from-scores", action="store_true",
                         help="skip data collection/scoring; just rebuild the map "
                              "from an existing Data/friction_scores.csv")
    args = parser.parse_args()
 
    DATA_DIR.mkdir(exist_ok=True)
    _ensure_census_csv()
 
    if not args.from_scores:
        if not os.environ.get("ORS_API_KEY"):
            print("WARNING: ORS_API_KEY is not set. clinic_Locs.py and travel_times.py "
                  "call the OpenRouteService API and will fail without it.\n")
        step_clinics(args.force)
        step_travel_times(args.force)
        step_friction_score(args.force)
    elif not _have("friction_scores.csv"):
        raise SystemExit("--from-scores was passed but Data/friction_scores.csv doesn't exist yet.")
 
    map_html = step_build_map(args.force)
 
    dest = ROOT / "index.html"
    shutil.copyfile(map_html, dest)
    print(f"\nCopied {map_html.relative_to(ROOT)} -> index.html")
    print("Commit index.html and push -- GitHub Pages / Netlify / Cloudflare Pages "
          "will serve it from the repo root with no extra config.")
 
 
if __name__ == "__main__":
    main()
 