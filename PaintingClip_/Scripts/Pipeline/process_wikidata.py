import requests
import pandas as pd
from openpyxl.utils import get_column_letter
import time
import logging

# ---------------- Setup Logging ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- User Parameters ----------------
total_limit = 100000         # Total number of paintings to retrieve
chunk_size = 1000          # Records to retrieve per chunk
sitelinks_threshold = 1    # Minimum number of sitelinks (notability proxy)

max_retries = 5            # Maximum number of retries for a query
base_sleep = 5             # Base sleep time (seconds) for retries
subchunk_size = 200        # Maximum number of painting IDs per aggregated query

# ---------------- Function to query Wikidata with retries (exponential backoff) ----------------
def query_wikidata(query):
    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "MyPythonSPARQLClient/0.1 (xlct43@durham.ac.uk)",
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    attempt = 0
    sleep_seconds = base_sleep
    while attempt < max_retries:
        try:
            logging.info(f"Sending query to Wikidata (attempt {attempt+1}/{max_retries})...")
            response = requests.post(url, headers=headers, data={'query': query})
            if response.status_code == 429:
                logging.warning("Rate limit encountered (HTTP 429). Sleeping for %s seconds...", sleep_seconds)
                time.sleep(sleep_seconds)
                attempt += 1
                sleep_seconds *= 2
                continue
            if response.status_code == 500:
                logging.warning("Internal Server Error (HTTP 500) encountered. Sleeping for %s seconds and retrying...", sleep_seconds)
                time.sleep(sleep_seconds)
                attempt += 1
                sleep_seconds *= 2
                continue
            if response.status_code == 414:
                logging.error("URI Too Long (HTTP 414) encountered. Your query is too large.")
                raise Exception("Query string is too long.")
            response.raise_for_status()
            logging.info("Query successful.")
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error("HTTP error encountered: %s", e)
            attempt += 1
            time.sleep(sleep_seconds)
            sleep_seconds *= 2
    raise Exception("Failed to retrieve data from Wikidata after several attempts.")

# ---------------- Function: Get Basic Records ----------------
def get_basic_records(offset, chunk_size, threshold):
    basic_query = f"""
    SELECT ?painting ?paintingLabel ?creator ?creatorLabel ?inception ?wikipedia_url ?linkCount WHERE {{
      ?painting wdt:P31 wd:Q3305213.
      ?painting wikibase:sitelinks ?linkCount.
      FILTER(?linkCount >= {threshold}).
      OPTIONAL {{ ?painting wdt:P170 ?creator. }}
      OPTIONAL {{ ?painting wdt:P571 ?inception. }}
      OPTIONAL {{
        ?paintingArticle schema:about ?painting ;
                         schema:inLanguage "en" ;
                         schema:isPartOf <https://en.wikipedia.org/> .
        BIND(?paintingArticle AS ?wikipedia_url)
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    ORDER BY DESC(?linkCount)
    LIMIT {chunk_size}
    OFFSET {offset}
    """
    data = query_wikidata(basic_query)
    bindings = data.get("results", {}).get("bindings", [])
    basic_records = {}
    painting_ids = []
    for item in bindings:
        pid = item.get("painting", {}).get("value", "")
        if pid:
            basic_records[pid] = {
                "Painting ID": pid,
                "Title": item.get("paintingLabel", {}).get("value", ""),
                "Creator ID": item.get("creator", {}).get("value", ""),
                "Creator": item.get("creatorLabel", {}).get("value", ""),
                "Inception": item.get("inception", {}).get("value", ""),
                "Wikipedia URL": item.get("wikipedia_url", {}).get("value", ""),
                "Link Count": item.get("linkCount", {}).get("value", "")
            }
            painting_ids.append(pid)
    return basic_records, painting_ids

# ---------------- Function: Get Aggregated Fields in Subchunks ----------------
def get_aggregated_fields(painting_ids):
    agg_records = {}
    for i in range(0, len(painting_ids), subchunk_size):
        sub_ids = painting_ids[i:i+subchunk_size]
        values_clause = "VALUES ?painting { " + " ".join(f"<{pid}>" for pid in sub_ids) + " }"
        agg_query = f"""
        SELECT ?painting 
               (GROUP_CONCAT(DISTINCT ?depictsLabel; separator=", ") AS ?depictsAggregated)
               (GROUP_CONCAT(DISTINCT ?movementLabel; separator=", ") AS ?movements)
               (GROUP_CONCAT(DISTINCT ?movement; separator=", ") AS ?movementIDs)
        WHERE {{
          {values_clause}
          OPTIONAL {{
              ?painting wdt:P180 ?depicts.
              ?depicts rdfs:label ?depictsLabel.
              FILTER(LANG(?depictsLabel) = "en")
          }}
          OPTIONAL {{
              ?painting wdt:P135 ?movement.
              ?movement rdfs:label ?movementLabel.
              FILTER(LANG(?movementLabel) = "en")
          }}
        }}
        GROUP BY ?painting
        """
        sub_agg_data = query_wikidata(agg_query)
        sub_bindings = sub_agg_data.get("results", {}).get("bindings", [])
        for item in sub_bindings:
            pid = item.get("painting", {}).get("value", "")
            if pid:
                agg_records[pid] = {
                    "Depicts": item.get("depictsAggregated", {}).get("value", ""),
                    "Movements": item.get("movements", {}).get("value", ""),
                    "Movement IDs": item.get("movementIDs", {}).get("value", "")
                }
    return agg_records

# ---------------- Function: Merge Basic and Aggregated Records ----------------
def merge_records(basic_records, agg_records):
    merged = []
    for pid, basic in basic_records.items():
        if pid in agg_records:
            basic.update(agg_records[pid])
        else:
            basic["Depicts"] = ""
            basic["Movements"] = ""
            basic["Movement IDs"] = ""
        merged.append(basic)
    return merged

# ---------------- Function: Process Record Fields ----------------
def process_record_fields(records):
    for rec in records:
        pid = rec.get("Painting ID", "")
        if pid:
            qid = pid.rstrip("/").split("/")[-1]
            rec["File Name"] = f"{qid}_0.png"
        else:
            rec["File Name"] = ""
        inception = rec.get("Inception", "")
        if inception and len(inception) >= 4:
            try:
                rec["Year"] = int(inception[:4])
            except ValueError:
                rec["Year"] = None
        else:
            rec["Year"] = None
    return records

# ---------------- Main Loop ----------------
logging.info(f"Starting retrieval of metadata for up to {total_limit} paintings...")
logging.info(f"Chunk size: {chunk_size}, Sitelinks threshold: >= {sitelinks_threshold}")

all_results = []
offset = 0

try:
    while len(all_results) < total_limit:
        logging.info(f"Querying basic records with OFFSET {offset} ...")
        basic_records, painting_ids = get_basic_records(offset, chunk_size, sitelinks_threshold)
        if not basic_records:
            logging.info("No more basic records returned; ending pagination.")
            break
        logging.info(f"Retrieved {len(basic_records)} unique basic records.")
    
        agg_records = get_aggregated_fields(painting_ids)
        merged_records = merge_records(basic_records, agg_records)
    
        # Append new records, avoiding duplicates.
        existing_ids = {r["Painting ID"] for r in all_results}
        for record in merged_records:
            if record["Painting ID"] not in existing_ids:
                all_results.append(record)
    
        offset += chunk_size
        if len(all_results) >= total_limit:
            logging.info("Reached the total desired number of records.")
            break
        time.sleep(1)  # Throttle between chunks
except KeyboardInterrupt:
    logging.info("Process interrupted by user.")

logging.info(f"Collected {len(all_results)} rows from the queries.")

all_results = process_record_fields(all_results)

# ---------------- Create DataFrame and Reorder Columns ----------------
# Final desired order:
# Title, File Name, Creator, Movements, Depicts, Year, Wikipedia URL, Link Count, Painting ID, Creator ID, Movement IDs
final_order = [
    "Title", "File Name", "Creator", "Movements", "Depicts", "Year",
    "Wikipedia URL", "Link Count", "Painting ID", "Creator ID", "Movement IDs"
]

df = pd.DataFrame(all_results)
for col in final_order:
    if col not in df.columns:
        df[col] = ""
df = df[final_order]

df["Link Count"] = pd.to_numeric(df["Link Count"], errors='coerce')
df.sort_values(by="Link Count", ascending=False, inplace=True)

logging.info("DataFrame created. Number of unique records: %s", len(df))

# ---------------- Write DataFrame to Excel ----------------
output_filename = "paintings2.xlsx"
logging.info(f"Writing data to Excel file '{output_filename}'...")

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Paintings')
    worksheet = writer.sheets['Paintings']
    # Adjust column widths based on maximum content length, capped at 60.
    for i, column in enumerate(df.columns, 1):
        max_length = max(df[column].astype(str).map(len).max(), len(column))
        adjusted_width = max_length + 2  # Extra space for readability
        if adjusted_width > 40:
            adjusted_width = 40
        worksheet.column_dimensions[get_column_letter(i)].width = adjusted_width

logging.info(f"Excel file saved to '{output_filename}'.")
