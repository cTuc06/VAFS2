import pandas as pd
import openrouteservice
from openrouteservice import convert

API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjU4ZDAwM2Q0YTc2YjQwY2I4NDYzOTI4MDkwYTEzZjcwIiwiaCI6Im11cm11cjY0In0="   # <-- put your ORS key here
client = openrouteservice.Client(key=API_KEY)

# Vaccination sites from your document
sites = [
    ("Cheverly Health Center", "3003 Hospital Drive, Cheverly, MD 20785", "Prince George's County"),
    ("Hyattsville Health Center", "1401 E. University Blvd, Suite 201, Hyattsville, MD 20783", "Prince George's County"),
    ("Germantown Health Center", "12900 Middlebrook Road, Germantown, MD 20874", "Montgomery County"),
    ("Silver Spring Health Center", "8630 Fenton Street, Silver Spring, MD 20910", "Montgomery County"),
    ("Dennis Ave Health Center", "2000 Dennis Ave, Silver Spring, MD 20902", "Montgomery County"),
    ("Eastern Health District Clinic", "1200 E. Fayette St, Baltimore, MD 21202", "Baltimore City"),
    ("East Baltimore Medical Center", "1000 E. Eager St, Baltimore, MD 21202", "Baltimore City"),
    ("Parole Health Center", "1950 Drew Street, Annapolis, MD 21401", "Anne Arundel County"),
    ("Glen Burnie Health Center", "416 A Street SW, Glen Burnie, MD 21061", "Anne Arundel County"),
    ("Magothy Health Center", "2501 Mountain Road, Pasadena, MD 21122", "Anne Arundel County"),
    ("Baltimore County Drumcastle", "6401 York Rd, Baltimore, MD 21212", "Baltimore County"),
    ("Dundalk Health Center", "7700 Dunmanway, Dundalk, MD 21222", "Baltimore County"),
    ("Laurel Clinic", "14207 Park Center Dr, Laurel, MD 20707", "Prince George's County"),
]

# Geocode any address or county

def geocode_location(text):
    result = client.pelias_search(text=text)

    if "features" not in result or len(result["features"]) == 0:
        raise ValueError(f"Could not geocode: {text}")
    
    coords = result["features"][0]["geometry"]["coordinates"]

    return coords[1], coords[0]  # lat, lon



# Compute distances

rows = []

for name, address, county in sites:
    print(f"Processing: {name}")

    # Geocode site
    site_lat, site_lon = geocode_location(address)

    # Geocode county center
    county_lat, county_lon = geocode_location(f"{county}, Maryland")

    # ORS directions
    route = client.directions(
        coordinates=[[site_lon, site_lat], [county_lon, county_lat]],
        profile="driving-car"
    )

    dist_m = route["routes"][0]["summary"]["distance"]
    dur_s = route["routes"][0]["summary"]["duration"]

    rows.append({
        "name": name,
        "address": address,
        "county": county,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "county_lat": county_lat,
        "county_lon": county_lon,
        "distance_meters": dist_m,
        "distance_miles": dist_m / 1609.34,
        "duration_minutes": dur_s / 60
    })


# Save output

df = pd.DataFrame(rows)
df.to_csv("Data\\md_vaccination_site_distances.csv", index=False)

print("Saved Data\\md_vaccination_site_distances.csv")


