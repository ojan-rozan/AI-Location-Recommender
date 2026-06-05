# Google Maps Places scraper
import os
import csv
import json
import time
import sys
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# project root supaya bisa import src.* dari mana aja
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import supabase_io as sio

# load the api key from .env file
load_dotenv(ROOT / ".env")
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# settings
base_url = "https://places.googleapis.com/v1"
folder_output = ROOT / "data" / "raw"
category = "cafe"
places_per_kecamatan = 50

# list of fields
detail_places_field = "id,displayName,formattedAddress,addressComponents,location,types,primaryType,primaryTypeDisplayName,internationalPhoneNumber,nationalPhoneNumber,websiteUri,googleMapsUri,regularOpeningHours,rating,userRatingCount,priceLevel,reviews,businessStatus"
search_field = "places.id,places.displayName,nextPageToken"


# text search api call
def search_places(query, page_token=None):
    # build the request
    url = base_url + "/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": search_field,
    }
    body = {
        "textQuery": query,
        "pageSize": 20,
        "regionCode": "ID",
        "languageCode": "id",
    }

    # add page token
    if page_token:
        body["pageToken"] = page_token

    # send the request, return empty dict if fail
    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        if response.status_code != 200:
            print("search error:", response.status_code)
            return {}
        return response.json()
    except Exception as e:
        print("search exception:", e)
        return {}


# place detail api
def get_place_detail(place_id):
    url = base_url + "/places/" + place_id
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": detail_places_field,
    }
    params = {"languageCode": "id", "regionCode": "ID"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            print("detail error:", response.status_code)
            return {}
        return response.json()
    except Exception as e:
        print("detail exception:", e)
        return {}


# cleaning opening hours
def clean_opening_hours(hours_data):
    if not hours_data:
        return ""

    # create in list
    weekday_list = hours_data.get("weekdayDescriptions", [])
    if not weekday_list:
        return ""

    # join with " | "
    return " | ".join(weekday_list)


# clean review list
def clean_reviews(reviews_data):
    if not reviews_data:
        return ""

    clean_list = []
    for review in reviews_data:
        # get review text
        text_obj = review.get("text")
        if isinstance(text_obj, dict):
            review_text = text_obj.get("text", "")
        else:
            review_text = text_obj or ""

        # get author
        author_obj = review.get("authorAttribution", {})
        author_name = author_obj.get("displayName", "")

        list_review = {
            "author": author_name,
            "rating_review": review.get("rating"),
            "text": review_text,
            "relative_time": review.get("relativePublishTimeDescription", ""),
            "publish_time": review.get("publishTime", ""),
        }
        clean_list.append(list_review)

    # save as json string so we can put it in csv
    return json.dumps(clean_list, ensure_ascii=False)


# get all record
def build_one_record(detail, kota, kecamatan):
    # place id
    place_id = detail.get("id", "")

    # place name
    name_obj = detail.get("displayName")
    if isinstance(name_obj, dict):
        place_name = name_obj.get("text", "")
    else:
        place_name = name_obj or ""

    # address
    address = detail.get("formattedAddress", "")

    # latitude longitude
    location = detail.get("location", {})
    if location is None:
        location = {}
    latitude = location.get("latitude")
    longitude = location.get("longitude")

    # primary type display name
    primary_type_obj = detail.get("primaryTypeDisplayName")
    if isinstance(primary_type_obj, dict):
        primary_type = primary_type_obj.get("text", "")
    else:
        primary_type = str(detail.get("primaryType", ""))

    # all types
    types_list = detail.get("types", [])
    if types_list is None:
        types_list = []
    all_types = ",".join(types_list)

    # phone numbers
    phone_intl = detail.get("internationalPhoneNumber", "")
    phone_national = detail.get("nationalPhoneNumber", "")

    # website and google maps link
    website = detail.get("websiteUri", "")
    google_maps_url = detail.get("googleMapsUri", "")

    # rating and total reviews count
    rating = detail.get("rating")
    total_reviews = detail.get("userRatingCount")

    # price level
    price_level = str(detail.get("priceLevel", ""))

    # business status
    business_status = detail.get("businessStatus", "")

    # opening hours and reviews
    opening_hours = clean_opening_hours(detail.get("regularOpeningHours"))
    list_reviews = clean_reviews(detail.get("reviews"))

    # put everything in one dict
    record = {
        "place_id": place_id,
        "name": place_name,
        "kota": kota,
        "kecamatan": kecamatan,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "primary_type": primary_type,
        "types": all_types,
        "phone_intl": phone_intl,
        "phone_national": phone_national,
        "website": website,
        "google_maps_url": google_maps_url,
        "rating": rating,
        "total_reviews": total_reviews,
        "price_category": price_level,
        "business_status": business_status,
        "opening_hours": opening_hours,
        "list_reviews": list_reviews,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
    }
    return record


# find all places in one search
def find_place_ids(category, kota, kecamatan, target):
    # build the search query
    query = category + " di Kecamatan " + kecamatan + ", " + kota
    print("  searching:", query)

    found_ids = []
    page_token = None

    # looping to get 60 places
    for page_number in range(3):
        # stop if we already have enough
        if len(found_ids) >= target:
            break

        # call the api
        response = search_places(query, page_token)
        time.sleep(0.5)

        # get list of places from response
        places_list = response.get("places", [])
        for place in places_list:
            pid = place.get("id")
            # only add if not duplicate
            if pid and pid not in found_ids:
                found_ids.append(pid)
                if len(found_ids) >= target:
                    break

        # get next page token, stop if no more pages
        page_token = response.get("nextPageToken")
        if not page_token:
            break

        time.sleep(2)

    # return 
    return found_ids[:target]


# scrape detail place for one kecamatan
def scrape_one_kecamatan(category, kota, kecamatan, target, already_scraped):
    records = []

    # find all place ids
    place_ids = find_place_ids(category, kota, kecamatan, target)
    print("  found", len(place_ids), "place ids")

    # get detail for each place
    counter = 0
    for pid in place_ids:
        counter = counter + 1

        # skip if already scraped before
        if pid in already_scraped:
            print("  [" + str(counter) + "] SKIP (already scraped)")
            continue

        # call detail api
        detail = get_place_detail(pid)
        time.sleep(0.5)

        # skip if api failed
        if not detail:
            print("  [" + str(counter) + "] FAILED")
            continue

        # build the record and add to list
        record = build_one_record(detail, kota, kecamatan)
        records.append(record)
        already_scraped.add(pid)

        # count reviews
        review_str = record.get("list_reviews", "")
        if review_str:
            review_count = len(json.loads(review_str))
        else:
            review_count = 0

        # print progress
        print("  [" + str(counter) + "]", record["name"],
              "| rating=", record["rating"],
              "| reviews=", review_count)

    return records


# save csv
def save_to_csv(records, path):
    if not records:
        return

    # get column names from first record
    columns = list(records[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print("saved csv:", path, "(", len(records), "rows)")


# save json
def save_to_json(records, path):
    # convert list_reviews string back to real list
    new_records = []

    for record in records:
        record_copy = dict(record)
        review_str = record_copy.get("list_reviews", "")
        if review_str:
            record_copy["reviews"] = json.loads(review_str)
        else:
            record_copy["reviews"] = []

        # remove the old string field
        record_copy.pop("list_reviews", None)
        new_records.append(record_copy)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)
    print("saved json:", path, "(", len(records), "rows)")


# load existing checkpoint
def load_existing_data(path):
    # if no checkpoint, return empty
    if not os.path.exists(path):
        return [], set()

    try:
        with open(path, encoding="utf-8") as f:
            old_data = json.load(f)

        records = []
        scraped_ids = set()

        for item in old_data:
            # turn reviews list back into string
            item_copy = dict(item)
            reviews_list = item_copy.pop("reviews", [])
            item_copy["list_reviews"] = json.dumps(reviews_list, ensure_ascii=False)
            records.append(item_copy)

            # remember which ids we already have
            pid = item.get("place_id", "")
            if pid:
                scraped_ids.add(pid)

        return records, scraped_ids
    except Exception as e:
        print("failed to load existing:", e)
        return [], set()


# main function
def main():
    # check if api key exists
    if not API_KEY:
        print("GOOGLE_MAPS_API_KEY not found in .env")
        return

    # read kecamatan dari Supabase (fallback ke CSV lokal)
    df = sio.read_df(sio.KECAMATAN_REF)
    if df.empty:
        local = ROOT / "data" / "helper" / "kecamatan_jakarta.csv"
        if local.exists():
            df = pd.read_csv(local)
        else:
            print("kecamatan_ref kosong di Supabase & CSV lokal gak ada")
            return
    print("loaded", len(df), "kecamatan")

    # make sure output folder exists
    os.makedirs(folder_output, exist_ok=True)

    # output file names
    file_csv = folder_output / f"places_{category}.csv"
    file_json = folder_output / f"places_{category}.json"
    file_checkpoint = folder_output / f"checkpoint_{category}.json"

    # try to load checkpoint
    all_records, already_scraped = load_existing_data(file_checkpoint)
    if all_records:
        print("resume:", len(all_records), "records,", len(already_scraped), "place ids")

    # print plan
    print("=" * 50)
    print("category    :", category)
    print("kecamatan   :", len(df))
    print("per kec     :", places_per_kecamatan)
    print("max target  :", len(df) * places_per_kecamatan)
    print("=" * 50)

    # loop each kecamatan
    counter = 0
    try:
        for index, row in df.iterrows():
            counter = counter + 1
            kota = row["kab_kota"]
            kecamatan = row["kecamatan"]
            print("\n[" + str(counter) + "/" + str(len(df)) + "]", kota, "/", kecamatan)

            # scrape one kecamatan
            try:
                new_records = scrape_one_kecamatan(
                    category, kota, kecamatan,
                    places_per_kecamatan, already_scraped
                )
                all_records.extend(new_records)
            except Exception as e:
                print("  kecamatan error:", e)

            # save checkpoint every 5 kecamatan
            if counter % 5 == 0:
                save_to_json(all_records, file_checkpoint)
                print("  checkpoint saved")

    except KeyboardInterrupt:
        print("\nstopped by user, saving partial data")
    except Exception as e:
        print("\nfatal error:", e)

    # save final results
    if all_records:
        save_to_csv(all_records, file_csv)
        save_to_json(all_records, file_json)
        save_to_json(all_records, file_checkpoint)

        # >>> upload hasil scraping ke Supabase <<<
        try:
            sio.upload_df(sio.RAW_CAFES, pd.DataFrame(all_records))
        except Exception as e:
            print("  ⚠️  gagal upload ke Supabase:", e)

    print("\ndone. total:", len(all_records), "records")


# run the script
if __name__ == "__main__":
    main()