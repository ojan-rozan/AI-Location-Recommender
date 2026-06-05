import os
import csv
import json
import time
import sys
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# project root supaya bisa import src.* dari mana aja
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import supabase_io as sio

# config
folder_output = ROOT / "data" / "raw"

# Jakarta mainland bbox (exclude Kepulauan Seribu)
bbox = (-6.37, 106.70, -6.08, 106.97)

overpass_mirrors = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

poi_categories = {
    "office": {
        "building": ["office"],
        "office": ["yes", "company", "coworking", "coworking_space",
                   "government", "lawyer", "accountant", "consulting",
                   "advertising_agency", "telecommunication"],
    },
    "mall": {
        "shop": ["mall"],
        "building": ["retail"],
    },
    "transit": {
        "railway": ["station", "halt", "tram_stop"],
        "highway": ["bus_stop"],
        "station": ["subway", "light_rail"],
        "amenity": ["bus_station", "ferry_terminal"],
    },
    "school": {
        "amenity": ["school", "university", "college", "kindergarten"],
    },
}

min_expected = {
    "office": 100,
    "mall": 20,
    "transit": 100,
    "school": 100,
}

# query for retrieve data
def build_query(tags, bbox):
    south, west, north, east = bbox
    bbox_str = str(south) + "," + str(west) + "," + str(north) + "," + str(east)

    parts = []
    for key, values in tags.items():
        if isinstance(values, str):
            values = [values]
        for v in values:
            parts.append('node["' + key + '"="' + v + '"](' + bbox_str + ');')
            parts.append('way["' + key + '"="' + v + '"](' + bbox_str + ');')

    query = """
        [out:json][timeout:180];
        (
        """ + " ".join(parts) + """
        );
        out center tags;
    """
    return query


def fetch_overpass(tags, bbox, min_expected=10):
    query = build_query(tags, bbox)

    for attempt in range(3):
        for mirror in overpass_mirrors:
            try:
                print("  fetch from", mirror[:42], "(attempt", str(attempt + 1) + ")")
                response = requests.post(
                    mirror, data={"data": query}, timeout=300
                )
                response.raise_for_status()
                data = response.json()
                count = len(data.get("elements", []))

                if count == 0:
                    print("    got 0 elements, try next mirror")
                    time.sleep(3)
                    continue

                if count < min_expected:
                    print("    got", count, "elements (< " + str(min_expected) + ", maybe partial)")
                else:
                    print("    got", count, "elements OK")

                return data
            except Exception as e:
                print("    fail:", type(e).__name__ + ":", str(e)[:80])
                time.sleep(3)

        print("    retry round", str(attempt + 1) + "/" + str(3))
        time.sleep(10)

    return None

# Parsing latitude and longitude
def get_lat_lng(element):
    if element.get("type") == "node":
        return element.get("lat"), element.get("lon")
    center = element.get("center", {})
    return center.get("lat"), center.get("lon")


def build_one_record(element, category):
    """Extract field natural OSM, gak dipaksa match Google Maps."""
    tags = element.get("tags", {})

    lat, lng = get_lat_lng(element)
    if lat is None or lng is None:
        return None

    # Detect main tag yang match (key=value yang relevan)
    main_tag = ""
    for k in ["amenity", "shop", "building", "office", "railway",
              "highway", "station"]:
        if tags.get(k):
            main_tag = k + "=" + tags[k]
            break

    record = {
        # Identifier
        "osm_type": element.get("type", ""),
        "osm_id": element.get("id", ""),

        # Category & tag detail
        "category": category,
        "main_tag": main_tag,

        # Name & branding
        "name": tags.get("name", ""),
        "name_id": tags.get("name:id", ""),
        "name_en": tags.get("name:en", ""),
        "brand": tags.get("brand", ""),
        "operator": tags.get("operator", ""),

        # Location
        "latitude": lat,
        "longitude": lng,

        # Address
        "addr_street": tags.get("addr:street", ""),
        "addr_housenumber": tags.get("addr:housenumber", ""),
        "addr_suburb": tags.get("addr:suburb", ""),
        "addr_city": tags.get("addr:city", ""),
        "addr_postcode": tags.get("addr:postcode", ""),

        # Contact
        "phone": tags.get("contact:phone", "") or tags.get("phone", ""),
        "website": tags.get("contact:website", "") or tags.get("website", ""),
        "email": tags.get("contact:email", "") or tags.get("email", ""),

        # Operational
        "opening_hours": tags.get("opening_hours", ""),
        "wheelchair": tags.get("wheelchair", ""),
        "wifi": tags.get("internet_access", ""),

        # All raw tags (for reference)
        "all_tags": json.dumps(tags, ensure_ascii=False),

        # Meta
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    return record

def parse_to_records(data, category):
    records = []
    seen_coords = set()

    elements = data.get("elements", [])
    total = len(elements)

    for element in elements:
        record = build_one_record(element, category)
        if record is None:
            continue

        # Dedupe by rounded coord
        coord_key = (round(record["latitude"], 5), round(record["longitude"], 5))
        if coord_key in seen_coords:
            continue
        seen_coords.add(coord_key)

        records.append(record)

    print("  parsed", len(records), "POI (from", total, "raw elements)")
    return records

# save to csv
def save_to_csv(records, path):
    if not records:
        return

    columns = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print("  saved csv:", path, "(" + str(len(records)) + " rows)")

# save to json
def save_to_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("  saved json:", path, "(" + str(len(records)) + " rows)")

# load existig data
def load_existing_csv(path):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as e:
        print("  failed to load existing:", e)
        return []

# scraping
def scrape_one_category(category, tags, cat_index, cat_total):
    print("\n[" + str(cat_index) + "/" + str(cat_total) + "] " + category)

    out_csv = folder_output / f"osm_{category}_jakarta.csv"
    out_json = folder_output / f"osm_{category}_jakarta.json"

    # Skip kalau udah ada
    existing = load_existing_csv(out_csv)
    min_exp = min_expected.get(category, 10)
    if len(existing) >= min_exp:
        print("  skip,", len(existing), "POI already saved")
        return

    data = fetch_overpass(tags, bbox, min_expected=min_exp)
    if data is None:
        print("  failed for", category)
        return

    records = parse_to_records(data, category)
    if len(records) == 0:
        print("  empty result")
        return

    save_to_csv(records, out_csv)
    save_to_json(records, out_json)

    # upload hasil scraping ke Supabas
    try:
        sio.upload_df(sio.raw_poi(category), pd.DataFrame(records))
    except Exception as e:
        print("  ⚠️  gagal upload ke Supabase:", e)

# Main function
def main():
    os.makedirs(folder_output, exist_ok=True)

    print("output dir:", folder_output)
    print("bbox:", bbox)

    total_categories = len(poi_categories)
    cat_index = 0
    for category, tags in poi_categories.items():
        cat_index = cat_index + 1
        try:
            scrape_one_category(category, tags, cat_index, total_categories)
        except Exception as e:
            print("  category error (" + category + "):", e)

    print("\ndone")


if __name__ == "__main__":
    main()