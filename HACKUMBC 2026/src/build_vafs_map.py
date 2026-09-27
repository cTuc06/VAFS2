"""
build_vafs_map.py
==================
Phase 3: regenerates the VAFS choropleth dashboard (vafs_map.html) from a
friction_scores.csv (Phase 2's output). Self-contained: fetches county and
ZCTA boundary polygons from public GitHub-hosted GeoJSON, simplifies them,
merges in your friction scores, and writes a single-file HTML map that
works with no server and no API keys (Leaflet loaded from a CDN at view
time; no basemap tiles are used, since polygon-only choropleths don't need
one and it keeps the page self-contained).

Requires: pandas, shapely  (pip install pandas shapely --break-system-packages)
No API keys needed. Needs outbound internet access to raw.githubusercontent.com.

Usage:
    python build_vafs_map.py
Reads:
    Data/friction_scores.csv   (from friction_score.py)
Writes:
    Data/vafs_map.html
"""

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
from shapely.geometry import shape, mapping

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR = Path("Data")
SCORES_CSV = DATA_DIR / "friction_scores.csv"
OUT_HTML = DATA_DIR / "vafs_map.html"

# Public boundary sources (no auth needed). Swap these if you move beyond
# Maryland or want higher-resolution boundaries.
COUNTY_GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)
# ZCTA source is per-state; change "md_maryland" for other states, e.g.
# "va_virginia_zip_codes_geo.min.json".
ZCTA_GEOJSON_URL = (
    "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/"
    "master/md_maryland_zip_codes_geo.min.json"
)

COUNTY_SIMPLIFY_TOL = 0.003   # degrees; larger = coarser/smaller file
ZCTA_SIMPLIFY_TOL = 0.0008    # ZCTAs are smaller, so a finer tolerance
COORD_PRECISION = 4           # decimal places (~11m at this latitude)

# Suggested intervention per top-friction-driver label. Keys must match the
# `label` values used in friction_score.py's INDICATORS list exactly.
INTERVENTIONS = {"Avg. drive time to nearest clinics (min)": "Deploy mobile or pop-up vaccination clinics closer to this area.", "% of workers not driving to work (no car access proxy)": "Add clinic sites reachable by transit/walking, or cover rideshare costs.", "% employed (proxy for rigid work schedules)": "Offer evening and weekend clinic hours.", "% NOT working from home (proxy for low schedule flexibility)": "Bring vaccination to workplaces via employer partnerships.", "Mean commute time to work (min)": "Add appointment slots near transit hubs and commute corridors.", "% uninsured": "Run no-cost, walk-in clinics with insurance navigation support on site.", "% households earning < $25k": "Offer free transportation vouchers and no-cost clinic days.", "% households on SNAP / cash assistance": "Co-locate vaccination events with SNAP/WIC offices or food banks.", "% in service/production/transport occupations (hourly-job proxy)": "Offer workplace vaccination and extended/weekend hours for shift workers."}


# ---------------------------------------------------------------------------
# STEP 1 -- fetch boundary GeoJSON (cached locally after first run)
# ---------------------------------------------------------------------------

def fetch_json(url: str, cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return json.loads(data)


# ---------------------------------------------------------------------------
# STEP 2 -- simplify + round a geometry to keep the final file small
# ---------------------------------------------------------------------------

def simplify_geometry(geom_dict: dict, tolerance: float, precision: int) -> dict:
    geom = shape(geom_dict)
    simplified = geom.simplify(tolerance, preserve_topology=True)
    gj = mapping(simplified)

    def round_coords(coords):
        if isinstance(coords[0], (list, tuple)):
            return [round_coords(c) for c in coords]
        return [round(c, precision) for c in coords]

    gj["coordinates"] = round_coords(gj["coordinates"])
    return gj


# ---------------------------------------------------------------------------
# STEP 3 -- match friction_scores.csv rows to boundary features by GEO_ID
# ---------------------------------------------------------------------------

def build_feature_collection(scores: pd.DataFrame) -> dict:
    scores_by_geo_id = {row["GEO_ID"]: row for _, row in scores.iterrows()}
    county_geo_ids = [g for g in scores_by_geo_id if g.startswith("0500000US")]
    zcta_geo_ids = [g for g in scores_by_geo_id if "Z200US" in g]
    zcta_codes = {g: scores_by_geo_id[g]["region_code"] for g in zcta_geo_ids}

    counties = fetch_json(COUNTY_GEOJSON_URL, DATA_DIR / "_cache_counties.json")
    zctas = fetch_json(ZCTA_GEOJSON_URL, DATA_DIR / "_cache_zctas.json")

    features = []
    unmatched = []

    for feat in counties["features"]:
        gid = feat["properties"].get("GEO_ID")
        if gid in county_geo_ids:
            row = scores_by_geo_id[gid]
            features.append(_to_feature(row, gid, "county",
                                         simplify_geometry(feat["geometry"], COUNTY_SIMPLIFY_TOL, COORD_PRECISION)))
    matched_county_ids = {f["properties"]["geo_id"] for f in features}
    unmatched += [g for g in county_geo_ids if g not in matched_county_ids]

    zcta_code_to_geoid = {v: k for k, v in zcta_codes.items()}
    for feat in zctas["features"]:
        code = feat["properties"].get("ZCTA5CE10")
        if code in zcta_code_to_geoid:
            gid = zcta_code_to_geoid[code]
            row = scores_by_geo_id[gid]
            features.append(_to_feature(row, gid, "zcta",
                                         simplify_geometry(feat["geometry"], ZCTA_SIMPLIFY_TOL, COORD_PRECISION)))
    matched_zcta_ids = {f["properties"]["geo_id"] for f in features if f["properties"]["geo_type"] == "zcta"}
    unmatched += [g for g in zcta_geo_ids if g not in matched_zcta_ids]

    if unmatched:
        print("WARNING: no boundary polygon found for these regions "
              "(they exist in friction_scores.csv but won't be drawn on the map):")
        for gid in unmatched:
            name = scores_by_geo_id[gid]["region_name"]
            print(f"  {gid}  ({name})")

    return {"type": "FeatureCollection", "features": features}


def _to_feature(row: pd.Series, geo_id: str, geo_type: str, geometry: dict) -> dict:
    def cell(key):
        val = row.get(key, "")
        return "" if pd.isna(val) else val

    return {
        "type": "Feature",
        "properties": {
            "geo_id": geo_id,
            "name": row["region_name"],
            "geo_type": geo_type,
            "score": cell("friction_score"),
            "drivers": cell("top_friction_drivers"),
            "physical": cell("physical_access"),
            "temporal": cell("temporal_access"),
            "economic": cell("economic_admin"),
            "social": cell("social_structural"),
        },
        "geometry": geometry,
    }


# ---------------------------------------------------------------------------
# STEP 4 -- inject data into the HTML shell
# ---------------------------------------------------------------------------

LEAFLET_CSS = """.leaflet-pane,.leaflet-tile,.leaflet-marker-icon,.leaflet-marker-shadow,.leaflet-tile-container,.leaflet-pane > svg,.leaflet-pane > canvas,.leaflet-zoom-box,.leaflet-image-layer,.leaflet-layer{position:absolute;left:0;top:0;}.leaflet-container{overflow:hidden;}.leaflet-tile,.leaflet-marker-icon,.leaflet-marker-shadow{-webkit-user-select:none;-moz-user-select:none;user-select:none;-webkit-user-drag:none;}.leaflet-tile::selection{background:transparent;}.leaflet-safari .leaflet-tile{image-rendering:-webkit-optimize-contrast;}.leaflet-safari .leaflet-tile-container{width:1600px;height:1600px;-webkit-transform-origin:0 0;}.leaflet-marker-icon,.leaflet-marker-shadow{display:block;}.leaflet-container .leaflet-overlay-pane svg{max-width:none !important;max-height:none !important;}.leaflet-container .leaflet-marker-pane img,.leaflet-container .leaflet-shadow-pane img,.leaflet-container .leaflet-tile-pane img,.leaflet-container img.leaflet-image-layer,.leaflet-container .leaflet-tile{max-width:none !important;max-height:none !important;width:auto;padding:0;}.leaflet-container img.leaflet-tile{mix-blend-mode:plus-lighter;}.leaflet-container.leaflet-touch-zoom{-ms-touch-action:pan-x pan-y;touch-action:pan-x pan-y;}.leaflet-container.leaflet-touch-drag{-ms-touch-action:pinch-zoom;touch-action:none;touch-action:pinch-zoom;}.leaflet-container.leaflet-touch-drag.leaflet-touch-zoom{-ms-touch-action:none;touch-action:none;}.leaflet-container{-webkit-tap-highlight-color:transparent;}.leaflet-container a{-webkit-tap-highlight-color:rgba(51,181,229,0.4);}.leaflet-tile{filter:inherit;visibility:hidden;}.leaflet-tile-loaded{visibility:inherit;}.leaflet-zoom-box{width:0;height:0;-moz-box-sizing:border-box;box-sizing:border-box;z-index:800;}.leaflet-overlay-pane svg{-moz-user-select:none;}.leaflet-pane{z-index:400;}.leaflet-tile-pane{z-index:200;}.leaflet-overlay-pane{z-index:400;}.leaflet-shadow-pane{z-index:500;}.leaflet-marker-pane{z-index:600;}.leaflet-tooltip-pane{z-index:650;}.leaflet-popup-pane{z-index:700;}.leaflet-map-pane canvas{z-index:100;}.leaflet-map-pane svg{z-index:200;}.leaflet-vml-shape{width:1px;height:1px;}.lvml{behavior:url(#default#VML);display:inline-block;position:absolute;}.leaflet-control{position:relative;z-index:800;pointer-events:visiblePainted;pointer-events:auto;}.leaflet-top,.leaflet-bottom{position:absolute;z-index:1000;pointer-events:none;}.leaflet-top{top:0;}.leaflet-right{right:0;}.leaflet-bottom{bottom:0;}.leaflet-left{left:0;}.leaflet-control{float:left;clear:both;}.leaflet-right .leaflet-control{float:right;}.leaflet-top .leaflet-control{margin-top:10px;}.leaflet-bottom .leaflet-control{margin-bottom:10px;}.leaflet-left .leaflet-control{margin-left:10px;}.leaflet-right .leaflet-control{margin-right:10px;}.leaflet-fade-anim .leaflet-popup{opacity:0;-webkit-transition:opacity 0.2s linear;-moz-transition:opacity 0.2s linear;transition:opacity 0.2s linear;}.leaflet-fade-anim .leaflet-map-pane .leaflet-popup{opacity:1;}.leaflet-zoom-animated{-webkit-transform-origin:0 0;-ms-transform-origin:0 0;transform-origin:0 0;}svg.leaflet-zoom-animated{will-change:transform;}.leaflet-zoom-anim .leaflet-zoom-animated{-webkit-transition:-webkit-transform 0.25s cubic-bezier(0,0,0.25,1);-moz-transition:-moz-transform 0.25s cubic-bezier(0,0,0.25,1);transition:transform 0.25s cubic-bezier(0,0,0.25,1);}.leaflet-zoom-anim .leaflet-tile,.leaflet-pan-anim .leaflet-tile{-webkit-transition:none;-moz-transition:none;transition:none;}.leaflet-zoom-anim .leaflet-zoom-hide{visibility:hidden;}.leaflet-interactive{cursor:pointer;}.leaflet-grab{cursor:-webkit-grab;cursor:-moz-grab;cursor:grab;}.leaflet-crosshair,.leaflet-crosshair .leaflet-interactive{cursor:crosshair;}.leaflet-popup-pane,.leaflet-control{cursor:auto;}.leaflet-dragging .leaflet-grab,.leaflet-dragging .leaflet-grab .leaflet-interactive,.leaflet-dragging .leaflet-marker-draggable{cursor:move;cursor:-webkit-grabbing;cursor:-moz-grabbing;cursor:grabbing;}.leaflet-marker-icon,.leaflet-marker-shadow,.leaflet-image-layer,.leaflet-pane > svg path,.leaflet-tile-container{pointer-events:none;}.leaflet-marker-icon.leaflet-interactive,.leaflet-image-layer.leaflet-interactive,.leaflet-pane > svg path.leaflet-interactive,svg.leaflet-image-layer.leaflet-interactive path{pointer-events:visiblePainted;pointer-events:auto;}.leaflet-container{background:#ddd;outline-offset:1px;}.leaflet-container a{color:#0078A8;}.leaflet-zoom-box{border:2px dotted #38f;background:rgba(255,255,255,0.5);}.leaflet-container{font-family:"Helvetica Neue",Arial,Helvetica,sans-serif;font-size:12px;font-size:0.75rem;line-height:1.5;}.leaflet-bar{box-shadow:0 1px 5px rgba(0,0,0,0.65);border-radius:4px;}.leaflet-bar a{background-color:#fff;border-bottom:1px solid #ccc;width:26px;height:26px;line-height:26px;display:block;text-align:center;text-decoration:none;color:black;}.leaflet-bar a,.leaflet-control-layers-toggle{background-position:50% 50%;background-repeat:no-repeat;display:block;}.leaflet-bar a:hover,.leaflet-bar a:focus{background-color:#f4f4f4;}.leaflet-bar a:first-child{border-top-left-radius:4px;border-top-right-radius:4px;}.leaflet-bar a:last-child{border-bottom-left-radius:4px;border-bottom-right-radius:4px;border-bottom:none;}.leaflet-bar a.leaflet-disabled{cursor:default;background-color:#f4f4f4;color:#bbb;}.leaflet-touch .leaflet-bar a{width:30px;height:30px;line-height:30px;}.leaflet-touch .leaflet-bar a:first-child{border-top-left-radius:2px;border-top-right-radius:2px;}.leaflet-touch .leaflet-bar a:last-child{border-bottom-left-radius:2px;border-bottom-right-radius:2px;}.leaflet-control-zoom-in,.leaflet-control-zoom-out{font:bold 18px 'Lucida Console',Monaco,monospace;text-indent:1px;}.leaflet-touch .leaflet-control-zoom-in,.leaflet-touch .leaflet-control-zoom-out{font-size:22px;}.leaflet-control-layers{box-shadow:0 1px 5px rgba(0,0,0,0.4);background:#fff;border-radius:5px;}.leaflet-control-layers-toggle{background-image:url(images/layers.png);width:36px;height:36px;}.leaflet-retina .leaflet-control-layers-toggle{background-image:url(images/layers-2x.png);background-size:26px 26px;}.leaflet-touch .leaflet-control-layers-toggle{width:44px;height:44px;}.leaflet-control-layers .leaflet-control-layers-list,.leaflet-control-layers-expanded .leaflet-control-layers-toggle{display:none;}.leaflet-control-layers-expanded .leaflet-control-layers-list{display:block;position:relative;}.leaflet-control-layers-expanded{padding:6px 10px 6px 6px;color:#333;background:#fff;}.leaflet-control-layers-scrollbar{overflow-y:scroll;overflow-x:hidden;padding-right:5px;}.leaflet-control-layers-selector{margin-top:2px;position:relative;top:1px;}.leaflet-control-layers label{display:block;font-size:13px;font-size:1.08333em;}.leaflet-control-layers-separator{height:0;border-top:1px solid #ddd;margin:5px -10px 5px -6px;}.leaflet-default-icon-path{background-image:url(images/marker-icon.png);}.leaflet-container .leaflet-control-attribution{background:#fff;background:rgba(255,255,255,0.8);margin:0;}.leaflet-control-attribution,.leaflet-control-scale-line{padding:0 5px;color:#333;line-height:1.4;}.leaflet-control-attribution a{text-decoration:none;}.leaflet-control-attribution a:hover,.leaflet-control-attribution a:focus{text-decoration:underline;}.leaflet-attribution-flag{display:inline !important;vertical-align:baseline !important;width:1em;height:0.6669em;}.leaflet-left .leaflet-control-scale{margin-left:5px;}.leaflet-bottom .leaflet-control-scale{margin-bottom:5px;}.leaflet-control-scale-line{border:2px solid #777;border-top:none;line-height:1.1;padding:2px 5px 1px;white-space:nowrap;-moz-box-sizing:border-box;box-sizing:border-box;background:rgba(255,255,255,0.8);text-shadow:1px 1px #fff;}.leaflet-control-scale-line:not(:first-child){border-top:2px solid #777;border-bottom:none;margin-top:-2px;}.leaflet-control-scale-line:not(:first-child):not(:last-child){border-bottom:2px solid #777;}.leaflet-touch .leaflet-control-attribution,.leaflet-touch .leaflet-control-layers,.leaflet-touch .leaflet-bar{box-shadow:none;}.leaflet-touch .leaflet-control-layers,.leaflet-touch .leaflet-bar{border:2px solid rgba(0,0,0,0.2);background-clip:padding-box;}.leaflet-popup{position:absolute;text-align:center;margin-bottom:20px;}.leaflet-popup-content-wrapper{padding:1px;text-align:left;border-radius:12px;}.leaflet-popup-content{margin:13px 24px 13px 20px;line-height:1.3;font-size:13px;font-size:1.08333em;min-height:1px;}.leaflet-popup-content p{margin:17px 0;margin:1.3em 0;}.leaflet-popup-tip-container{width:40px;height:20px;position:absolute;left:50%;margin-top:-1px;margin-left:-20px;overflow:hidden;pointer-events:none;}.leaflet-popup-tip{width:17px;height:17px;padding:1px;margin:-10px auto 0;pointer-events:auto;-webkit-transform:rotate(45deg);-moz-transform:rotate(45deg);-ms-transform:rotate(45deg);transform:rotate(45deg);}.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:white;color:#333;box-shadow:0 3px 14px rgba(0,0,0,0.4);}.leaflet-container a.leaflet-popup-close-button{position:absolute;top:0;right:0;border:none;text-align:center;width:24px;height:24px;font:16px/24px Tahoma,Verdana,sans-serif;color:#757575;text-decoration:none;background:transparent;}.leaflet-container a.leaflet-popup-close-button:hover,.leaflet-container a.leaflet-popup-close-button:focus{color:#585858;}.leaflet-popup-scrolled{overflow:auto;}.leaflet-oldie .leaflet-popup-content-wrapper{-ms-zoom:1;}.leaflet-oldie .leaflet-popup-tip{width:24px;margin:0 auto;-ms-filter:"progid:DXImageTransform.Microsoft.Matrix(M11=0.70710678,M12=0.70710678,M21=-0.70710678,M22=0.70710678)";filter:progid:DXImageTransform.Microsoft.Matrix(M11=0.70710678,M12=0.70710678,M21=-0.70710678,M22=0.70710678);}.leaflet-oldie .leaflet-control-zoom,.leaflet-oldie .leaflet-control-layers,.leaflet-oldie .leaflet-popup-content-wrapper,.leaflet-oldie .leaflet-popup-tip{border:1px solid #999;}.leaflet-div-icon{background:#fff;border:1px solid #666;}.leaflet-tooltip{position:absolute;padding:6px;background-color:#fff;border:1px solid #fff;border-radius:3px;color:#222;white-space:nowrap;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;pointer-events:none;box-shadow:0 1px 3px rgba(0,0,0,0.4);}.leaflet-tooltip.leaflet-interactive{cursor:pointer;pointer-events:auto;}.leaflet-tooltip-top:before,.leaflet-tooltip-bottom:before,.leaflet-tooltip-left:before,.leaflet-tooltip-right:before{position:absolute;pointer-events:none;border:6px solid transparent;background:transparent;content:"";}.leaflet-tooltip-bottom{margin-top:6px;}.leaflet-tooltip-top{margin-top:-6px;}.leaflet-tooltip-bottom:before,.leaflet-tooltip-top:before{left:50%;margin-left:-6px;}.leaflet-tooltip-top:before{bottom:0;margin-bottom:-12px;border-top-color:#fff;}.leaflet-tooltip-bottom:before{top:0;margin-top:-12px;margin-left:-6px;border-bottom-color:#fff;}.leaflet-tooltip-left{margin-left:-6px;}.leaflet-tooltip-right{margin-left:6px;}.leaflet-tooltip-left:before,.leaflet-tooltip-right:before{top:50%;margin-top:-6px;}.leaflet-tooltip-left:before{right:0;margin-right:-12px;border-left-color:#fff;}.leaflet-tooltip-right:before{left:0;margin-left:-12px;border-right-color:#fff;}@media print{.leaflet-control{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}"""

HTML_SHELL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Vaccine Access Friction Score — Maryland</title>
<style>
__LEAFLET_CSS__
:root {
  --bg: #f7f5f1; --panel: #ffffff; --ink: #1c1a17; --ink-dim: #6b665f;
  --border: #e3ddd2; --accent: #7a3b2e;
  box-sizing: border-box;
  padding-top: env(safe-area-inset-top, 0px);
  padding-bottom: env(safe-area-inset-bottom, 0px);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #17140f; --panel: #23201a; --ink: #f1ede4; --ink-dim: #a89f8f;
    --border: #3a352b; --accent: #e0a382;
  }
}
:root[data-theme="dark"] {
  --bg: #17140f; --panel: #23201a; --ink: #f1ede4; --ink-dim: #a89f8f;
  --border: #3a352b; --accent: #e0a382;
}
html { scroll-padding-top: env(safe-area-inset-top, 0px); }
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  height: 100%;
}
html, body { height: 100%; }
.app { display: flex; flex-direction: column; height: 100vh; }
header {
  padding: 14px 18px calc(10px + env(safe-area-inset-top, 0px));
  border-bottom: 1px solid var(--border); background: var(--panel);
}
header h1 { margin: 0; font-size: 1.15rem; letter-spacing: -0.01em; }
header p { margin: 4px 0 0; font-size: 0.82rem; color: var(--ink-dim); max-width: 70ch; }
.main { flex: 1; display: flex; min-height: 0; }
@media (max-width: 760px) { .main { flex-direction: column; } }
#map { flex: 1; min-height: 0; background: var(--bg); }
.panel {
  width: 320px; flex-shrink: 0; border-left: 1px solid var(--border);
  background: var(--panel); overflow-y: auto; padding: 16px;
}
@media (max-width: 760px) {
  .panel { width: auto; border-left: none; border-top: 1px solid var(--border); max-height: 42vh; }
}
.panel h2 { margin: 0 0 2px; font-size: 1rem; }
.panel .score-big { font-size: 2.2rem; font-weight: 700; margin: 6px 0 2px; }
.panel .score-label { font-size: 0.78rem; color: var(--ink-dim); margin-bottom: 14px; }
.domain-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 0.8rem; }
.domain-row .name { width: 92px; flex-shrink: 0; color: var(--ink-dim); }
.domain-row .bar-track { flex: 1; height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; }
.domain-row .bar-fill { height: 100%; background: var(--accent); }
.domain-row .val { width: 30px; text-align: right; flex-shrink: 0; }
.section-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-dim); margin: 18px 0 8px; }
.driver-item { padding: 8px 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; }
.driver-item .label { font-size: 0.8rem; font-weight: 600; }
.driver-item .rec { font-size: 0.78rem; color: var(--ink-dim); margin-top: 3px; }
.empty-hint { font-size: 0.85rem; color: var(--ink-dim); line-height: 1.5; }
.nodata-note { font-size: 0.8rem; color: var(--ink-dim); background: var(--bg); border: 1px solid var(--border); padding: 8px 10px; border-radius: 8px; }
.legend {
  background: var(--panel); padding: 8px 10px; border-radius: 8px;
  border: 1px solid var(--border); font-size: 0.75rem; color: var(--ink);
  line-height: 1.5;
}
.legend .row { display: flex; align-items: center; gap: 6px; }
.legend .swatch { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
footer {
  padding: 8px 18px calc(10px + env(safe-area-inset-bottom, 0px)); font-size: 0.72rem;
  color: var(--ink-dim); border-top: 1px solid var(--border); background: var(--panel);
}
.leaflet-container { background: var(--bg); }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>Vaccine Access Friction Score — Maryland (demo)</h1>
    <p>Composite 0–100 friction score per region, built from ACS DP03 (county &amp; ZCTA) data. Click a region for its domain breakdown and suggested interventions.</p>
  </header>
  <div class="main">
    <div id="map"></div>
    <div class="panel" id="panel">
      <div class="empty-hint">Click a region on the map to see its friction score, domain breakdown, top drivers, and suggested interventions.</div>
    </div>
  </div>
  <footer>
    Demo dataset: 7 MD counties + 6 Baltimore-area ZCTAs. Physical-access domain uses a vehicle-access proxy only (no live routing data in this demo). Social/structural domain is partial (hourly-job proxy only; no childcare or misinformation data available). Not for operational use.
  </footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const REGIONS = __REGIONS_JSON__;
const INTERVENTIONS = __INTERVENTIONS_JSON__;

const BINS = [20, 40, 60, 80, 100];
const COLORS = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'];
const NODATA_COLOR = '#c9c2b4';

function colorFor(score) {
  if (score === null || score === undefined || isNaN(score)) return NODATA_COLOR;
  for (let i = 0; i < BINS.length; i++) {
    if (score <= BINS[i]) return COLORS[i];
  }
  return COLORS[COLORS.length - 1];
}

function parseScore(raw) {
  const v = parseFloat(raw);
  return isNaN(v) ? null : v;
}

const map = L.map('map', { zoomControl: true, attributionControl: true, minZoom: 6, maxZoom: 13 });

function style(feature) {
  const score = parseScore(feature.properties.score);
  return {
    fillColor: colorFor(score), fillOpacity: 0.75,
    color: '#3a352b', weight: 1, opacity: 0.6,
  };
}

function highlight(e) {
  e.target.setStyle({ weight: 2.5, opacity: 1 });
  e.target.bringToFront();
}
function unhighlight(e) {
  geoLayer.resetStyle(e.target);
}

function parseDrivers(driversStr) {
  if (!driversStr) return [];
  return driversStr.split(';').map(s => s.trim()).filter(Boolean).map(seg => {
    const m = seg.match(/^(.*)\s\((\d+)\)$/);
    return m ? { label: m[1], score: m[2] } : { label: seg, score: '' };
  });
}

function showPanel(feature) {
  const p = feature.properties;
  const panel = document.getElementById('panel');
  const score = parseScore(p.score);

  if (score === null) {
    panel.innerHTML = `
      <h2>${p.name}</h2>
      <div class="nodata-note">No usable ACS data for this region (values suppressed or population too small to report) — shown gray on the map, excluded from the score.</div>`;
    return;
  }

  const domains = [
    { key: 'physical', label: 'Physical' },
    { key: 'temporal', label: 'Temporal' },
    { key: 'economic', label: 'Economic' },
    { key: 'social', label: 'Social' },
  ];
  const domainRows = domains.map(d => {
    const v = parseScore(p[d.key]);
    const pct = v === null ? 0 : v;
    const display = v === null ? '—' : Math.round(v);
    return `<div class="domain-row">
      <span class="name">${d.label}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${pct}%"></span></span>
      <span class="val">${display}</span>
    </div>`;
  }).join('');

  const drivers = parseDrivers(p.drivers).slice(0, 3);
  const driverItems = drivers.map(d => {
    const rec = INTERVENTIONS[d.label] || 'Review this indicator locally to identify a targeted intervention.';
    return `<div class="driver-item">
      <div class="label">${d.label}</div>
      <div class="rec">→ ${rec}</div>
    </div>`;
  }).join('');

  panel.innerHTML = `
    <h2>${p.name}</h2>
    <div class="score-big">${Math.round(score)}</div>
    <div class="score-label">composite friction score (0–100)</div>
    <div class="section-title">Domain breakdown</div>
    ${domainRows}
    <div class="section-title">Top friction drivers</div>
    ${driverItems || '<div class="empty-hint">No individual drivers available.</div>'}
  `;
}

const geoLayer = L.geoJSON(REGIONS, {
  style: style,
  onEachFeature: function (feature, layer) {
    layer.on({
      mouseover: highlight,
      mouseout: unhighlight,
      click: function () { showPanel(feature); },
    });
  },
}).addTo(map);

map.fitBounds(geoLayer.getBounds(), { padding: [16, 16] });

const legend = L.control({ position: 'bottomright' });
legend.onAdd = function () {
  const div = L.DomUtil.create('div', 'legend');
  const labels = ['0–20', '20–40', '40–60', '60–80', '80–100'];
  let rows = '<div style="font-weight:600;margin-bottom:4px;">Friction score</div>';
  for (let i = 0; i < COLORS.length; i++) {
    rows += `<div class="row"><span class="swatch" style="background:${COLORS[i]}"></span>${labels[i]}</div>`;
  }
  rows += `<div class="row" style="margin-top:4px;"><span class="swatch" style="background:${NODATA_COLOR}"></span>No data</div>`;
  div.innerHTML = rows;
  return div;
};
legend.addTo(map);
</script>
</body>
</html>
"""


def render_html(feature_collection: dict) -> str:
    html = HTML_SHELL
    html = html.replace("__LEAFLET_CSS__", LEAFLET_CSS, 1)
    html = html.replace("__REGIONS_JSON__", json.dumps(feature_collection, separators=(",", ":")), 1)
    html = html.replace("__INTERVENTIONS_JSON__", json.dumps(INTERVENTIONS), 1)
    return html


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not SCORES_CSV.exists():
        raise SystemExit(f"{SCORES_CSV} not found -- run friction_score.py first.")

    scores = pd.read_csv(SCORES_CSV, dtype={"region_code": str})
    print(f"Loaded {len(scores)} regions from {SCORES_CSV}")

    fc = build_feature_collection(scores)
    print(f"Matched {len(fc['features'])} regions to boundary polygons")

    html = render_html(fc)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved {OUT_HTML} ({len(html)} bytes)")


if __name__ == "__main__":
    main()