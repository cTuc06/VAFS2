"""
travel_times.py
================
Phase 1, Step 5: Approximate travel times from Maryland county centroids
(census geography you actually have -- ACSDP5Y2024_DP03 is COUNTY level,
GEO_ID like "0500000US24003") to clinic locations (from clinic_Locs.py's
md_clinics.csv output), using the OpenRouteService (ORS) Matrix API.

Why the Matrix API instead of looping client.directions() like scales.py did:
  - One matrix call returns times/distances for MANY origin x destination
    pairs at once, instead of one HTTP request per pair.
  - ORS free tier limits (as of writing): max ~2500 origin x destination
    elements per matrix request, ~70 total locations. This script batches
    automatically to stay under that.

TRANSIT: ORS / OSRM / GraphHopper (the three "free routing API" options you
listed) only do car / bike / foot routing -- none of them do real transit
(bus/rail) routing out of the box, because that needs a GTFS feed and a
transit-aware engine (e.g. OpenTripPlanner, or a paid API like Google Routes).
Rather than fabricate transit numbers, this script leaves compute_transit_time()
as a clearly-marked stub. See the docstring on that function for your actual
free options (MTA Maryland GTFS, WMATA GTFS, OpenTripPlanner).

Usage:
    python travel_times.py

Outputs:
    county_centroids.json   -- cached geocoded county centroids (reused on reruns)
    travel_times.csv        -- long-format table: one row per county x clinic pair
    nearest_clinics.csv     -- for each county, its N closest clinics by drive time
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import openrouteservice
from openrouteservice.exceptions import ApiError

# ---------------------------------------------------------------------------
# CONFIG -- adjust paths/keys as needed
# ---------------------------------------------------------------------------

DATA_DIR = Path("Data")  # adjust to wherever your files actually live

CENSUS_CSV = DATA_DIR / "ACSDP5Y2024.DP03-Data.csv"
CLINICS_CSV = DATA_DIR / "md_vaccination_site_distances.csv"          # output of clinic_Locs.py
CENTROID_CACHE = DATA_DIR / "county_centroids.json"

OUT_TRAVEL_TIMES = DATA_DIR / "travel_times.csv"
OUT_NEAREST = DATA_DIR / "nearest_clinics.csv"

# !! Put your ORS key in an environment variable instead of hardcoding it.
#    (scales.py had a live key committed in plain text -- rotate that key.)
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjU4ZDAwM2Q0YTc2YjQwY2I4NDYzOTI4MDkwYTEzZjcwIiwiaCI6Im11cm11cjY0In0="

PROFILE = "driving-car"          # ORS profile: driving-car, cycling-regular, foot-walking
N_NEAREST = 5                     # how many nearest clinics to keep per county
MAX_ORIGINS_PER_CALL = 3          # keep origins x destinations under ORS's ~2500 element cap
REQUEST_PAUSE_SEC = 1.0           # be polite to the free-tier rate limit

# Maryland counties from your DP03 file. ORS's geocoder (Pelias) resolves
# "<County>, Maryland" fine for all of these, same pattern scales.py used
# for its "county center" geocoding.
# (Baltimore city is an independent city, not a county -- listed separately.)


# ---------------------------------------------------------------------------
# STEP 1 -- load & clean census data (county level)
# ---------------------------------------------------------------------------

def load_census_data(path: Path) -> pd.DataFrame:
    """Load and lightly clean the ACS DP03 county-level extract.

    Mirrors the column selection in census_data.py, but keeps it importable
    (no top-level print/side effects) so it can be reused here.
    """
    df = pd.read_csv(path, skiprows=[1])  # row 1 is the human-readable header dupe

    keep = [
        "DP03_0001E", "DP03_0002E", "DP03_0003E", "DP03_0004E", "DP03_0005E", "DP03_0007E",
        "DP03_0019E", "DP03_0020E", "DP03_0021E", "DP03_0022E", "DP03_0023E", "DP03_0024E", "DP03_0025E",
        "DP03_0051E", "DP03_0052E", "DP03_0053E", "DP03_0054E", "DP03_0055E", "DP03_0056E",
        "DP03_0057E", "DP03_0058E", "DP03_0059E", "DP03_0060E", "DP03_0061E", "DP03_0062E",
        "DP03_0063E", "DP03_0064E", "DP03_0065E",
        "DP03_0066E", "DP03_0067E", "DP03_0068E", "DP03_0069E", "DP03_0070E", "DP03_0071E",
        "DP03_0072E", "DP03_0073E", "DP03_0074E",
        "DP03_0095E", "DP03_0096E", "DP03_0097E", "DP03_0098E", "DP03_0099E", "DP03_0100E",
        "DP03_0101E", "DP03_0102E", "DP03_0103E", "DP03_0104E", "DP03_0105E", "DP03_0106E",
        "DP03_0107E", "DP03_0108E", "DP03_0109E", "DP03_0110E", "DP03_0111E", "DP03_0112E",
        "DP03_0113E", "DP03_0114E", "DP03_0115E", "DP03_0116E", "DP03_0117E",
    ]
    keep = ["GEO_ID", "NAME"] + [c for c in keep if c in df.columns]

    df_small = df[keep].copy()
    numeric_cols = [c for c in df_small.columns if c not in ("GEO_ID", "NAME")]
    df_small[numeric_cols] = df_small[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df_small.replace({-666666666: None, -999999999: None, -888888888: None}, inplace=True)
    df_small.fillna(0, inplace=True)

    # NAME looks like "Anne Arundel County, Maryland" -- split off a clean
    # "county_query" column we can hand straight to the geocoder.
    df_small["county_query"] = df_small["NAME"]
    return df_small


# ---------------------------------------------------------------------------
# STEP 2 -- load clinics (output of clinic_Locs.py)
# ---------------------------------------------------------------------------

def load_clinics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run clinic_Locs.py first to generate md_clinics.csv "
            "(it downloads clinic/hospital locations from OpenStreetMap)."
        )
    df = pd.read_csv(path)
    required = {"site_lat", "site_lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    # Prefer a human-readable name column if OSM gave us one; fall back to osmid.
    name_col = next((c for c in ["name", "amenity", "healthcare"] if c in df.columns), None)
    df["clinic_name"] = df[name_col] if name_col else df.get("osmid", df.index.astype(str))
    df["clinic_id"] = df.get("osmid", df.index).astype(str)

    df = df.dropna(subset=["site_lat", "site_lon"]).reset_index(drop=True)
    return df[["clinic_id", "clinic_name", "site_lat", "site_lon"]]


# ---------------------------------------------------------------------------
# STEP 3 -- geocode county centroids (cached)
# ---------------------------------------------------------------------------

def get_county_centroids(client: openrouteservice.Client, counties: list[str], cache_path: Path) -> dict:
    """Geocode each county name to (lat, lon), reusing a local cache so
    re-running the script doesn't re-spend API calls on the same 24 counties.
    """
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    for county in counties:
        if county in cache:
            continue
        print(f"Geocoding: {county}")
        try:
            result = client.pelias_search(text=county)
            features = result.get("features", [])
            if not features:
                print(f"  WARNING: could not geocode '{county}', skipping.")
                continue
            lon, lat = features[0]["geometry"]["coordinates"]
            cache[county] = {"county_lat": lat, "county_lon": lon}
        except ApiError as e:
            print(f"  ERROR geocoding '{county}': {e}")
        time.sleep(REQUEST_PAUSE_SEC)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))
    return cache


# ---------------------------------------------------------------------------
# STEP 4 -- bulk driving times/distances via ORS Matrix API
# ---------------------------------------------------------------------------

def compute_drive_matrix(
    client: openrouteservice.Client,
    origins: pd.DataFrame,   # columns: id, lat, lon
    destinations: pd.DataFrame,  # columns: id, lat, lon
) -> pd.DataFrame:
    """Return a long dataframe: origin_id, destination_id, duration_min, distance_miles.

    Batches origins so that (origins_in_batch * n_destinations) stays under
    ORS's free-tier matrix element cap.
    """
    dest_coords = list(zip(destinations["site_lon"], destinations["site_lat"]))
    n_dest = len(dest_coords)
    batch_size = max(1, MAX_ORIGINS_PER_CALL)

    rows = []
    for start in range(0, len(origins), batch_size):
        batch = origins.iloc[start:start + batch_size]
        origin_coords = list(zip(batch["county_lon"], batch["county_lat"]))
        locations = origin_coords + dest_coords
        n_origin = len(origin_coords)

        print(f"Matrix call: {n_origin} origins x {n_dest} destinations "
              f"({n_origin * n_dest} elements)")

        try:
            matrix = client.distance_matrix(
                locations=locations,
                profile=PROFILE,
                sources=list(range(0, n_origin)),
                destinations=list(range(n_origin, n_origin + n_dest)),
                metrics=["duration", "distance"],
                units="mi",
            )
        except ApiError as e:
            print(f"  ERROR on batch starting at {start}: {e}")
            time.sleep(REQUEST_PAUSE_SEC)
            continue

        durations = matrix["durations"]   # seconds, [origin][dest]
        distances = matrix["distances"]   # miles (per units="mi" above)

        for i, (_, o) in enumerate(batch.iterrows()):
            for j, (_, d) in enumerate(destinations.iterrows()):
                dur = durations[i][j]
                dist = distances[i][j]
                rows.append({
                    "origin_id": o["id"],
                    "destination_id": d["id"],
                    "duration_min": None if dur is None else round(dur / 60, 1),
                    "distance_miles": None if dist is None else round(dist, 2),
                })

        time.sleep(REQUEST_PAUSE_SEC)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TRANSIT -- honest stub, not fabricated numbers
# ---------------------------------------------------------------------------

def compute_transit_time():
    """Placeholder: none of OSRM / GraphHopper / ORS do real transit routing
    for free -- that needs a GTFS feed + a transit-aware router.

    Your actual free options, in order of effort:
      1. OpenTripPlanner (self-hosted, free) + GTFS feeds:
         - MTA Maryland GTFS:  https://www.mta.maryland.gov/developer-resources
         - WMATA GTFS (DC-adjacent counties): https://www.wmata.com/schedules/developers/gtfs.cfm
         Load both feeds + OSM street data into OTP, then call its
         GraphQL/REST API with mode=TRANSIT for real bus/rail itineraries.
      2. Google Routes API (Distance Matrix, mode=transit) -- accurate and
         easy, but requires a billing-enabled Google Cloud project (not free
         beyond a monthly credit).
      3. A crude heuristic (NOT recommended for anything you'll present as
         real data): multiply driving duration by a fixed multiplier (e.g.
         1.5-3x depending on urban vs. rural county) to approximate transit
         friction. Only use this as a rough placeholder and label it as such
         in any output/chart.
    """
    raise NotImplementedError(
        "Transit routing needs a GTFS-aware router (see docstring). "
        "Not implemented here to avoid producing fabricated numbers."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not ORS_API_KEY:
        raise SystemExit(
            "Set the ORS_API_KEY environment variable before running "
            "(don't hardcode it in the script)."
        )
    client = openrouteservice.Client(key=ORS_API_KEY)

    print("Loading census data...")
    census = load_census_data(CENSUS_CSV)

    print("Loading clinics...")
    clinics = load_clinics(CLINICS_CSV)
    print(f"  {len(clinics)} clinics loaded.")

    print("Geocoding county centroids (cached)...")
    counties = census["county_query"].tolist()
    centroid_cache = get_county_centroids(client, counties, CENTROID_CACHE)

    origins = pd.DataFrame([
        {"id": name, "county_lat": v["county_lat"], "county_lon": v["county_lon"]}
        for name, v in centroid_cache.items()
    ])
    destinations = clinics.rename(columns={"clinic_id": "id"})[["id", "site_lat", "site_lon"]]

    print("Computing drive-time matrix via ORS...")
    matrix = compute_drive_matrix(client, origins, destinations)

    # Attach human-readable clinic names + join census attributes back in.
    matrix = matrix.merge(
        clinics.rename(columns={"clinic_id": "destination_id"})[["destination_id", "clinic_name"]],
        on="destination_id", how="left",
    )
    matrix = matrix.merge(
        census.rename(columns={"county_query": "origin_id"}),
        on="origin_id", how="left",
    )

    matrix.to_csv(OUT_TRAVEL_TIMES, index=False)
    print(f"Saved {OUT_TRAVEL_TIMES} ({len(matrix)} rows).")

    nearest = (
        matrix.dropna(subset=["duration_min"])
        .sort_values(["origin_id", "duration_min"])
        .groupby("origin_id")
        .head(N_NEAREST)
    )
    nearest.to_csv(OUT_NEAREST, index=False)
    print(f"Saved {OUT_NEAREST} (top {N_NEAREST} nearest clinics per county).")


if __name__ == "__main__":
    main()