#!/usr/bin/env python3
"""
enrich_works_metadata.py
------------------------
Enrich works.json with bibliographic metadata (author, title, year, BibTeX)
by querying the Crossref API using DOIs.

Usage
-----
$ python enrich_works_metadata.py
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from tqdm import tqdm

# Configuration
CROSSREF_API_BASE = "https://api.crossref.org/works/"
CROSSREF_MAILTO = "samjmwaugh@gmail.com"  # Replace with your email
RATE_LIMIT_DELAY = 0.1  # Seconds between requests (be polite to API)
TIMEOUT = 10  # Request timeout in seconds

# File paths
ROOT = Path(__file__).resolve().parent
WORKS_FILE = ROOT / "works.json"
OUTPUT_FILE = ROOT / "works_enriched.json"


def clean_doi(doi_url: str) -> str:
    """Extract clean DOI from URL format."""
    if doi_url.startswith("https://doi.org/"):
        return doi_url.replace("https://doi.org/", "")
    return doi_url


def format_author_name(author: Dict) -> str:
    """Format author name from Crossref author object."""
    given = author.get("given", "")
    family = author.get("family", "")

    if given and family:
        return f"{given} {family}"
    elif family:
        return family
    else:
        return author.get("name", "Unknown Author")


def format_authors(authors: list) -> str:
    """Format multiple authors with proper separation."""
    if not authors:
        return "Unknown Author"

    formatted = [format_author_name(author) for author in authors]

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"


def generate_bibtex(metadata: Dict, work_id: str) -> str:
    """Generate BibTeX citation from Crossref metadata."""
    # Determine entry type
    if metadata.get("type") == "journal-article":
        entry_type = "article"
    elif metadata.get("type") == "book":
        entry_type = "book"
    elif metadata.get("type") == "book-chapter":
        entry_type = "incollection"
    elif metadata.get("type") == "proceedings-article":
        entry_type = "inproceedings"
    else:
        entry_type = "misc"

    # Extract year
    date_parts = metadata.get("published-print", {}).get("date-parts", [[]])
    if not date_parts or not date_parts[0]:
        date_parts = metadata.get("published-online", {}).get("date-parts", [[]])
    year = str(date_parts[0][0]) if date_parts and date_parts[0] else "n.d."

    # Build BibTeX
    bibtex_lines = [f"@{entry_type}{{{work_id},"]

    # Authors
    if metadata.get("author"):
        author_names = [format_author_name(author) for author in metadata["author"]]
        bibtex_lines.append(f'  author = "{" and ".join(author_names)}",')

    # Title
    title = metadata.get("title", ["Unknown Title"])[0]
    bibtex_lines.append(f'  title = "{{{title}}}",')

    # Year
    bibtex_lines.append(f'  year = "{year}",')

    # Journal/Book info
    if entry_type == "article" and metadata.get("container-title"):
        journal = metadata["container-title"][0]
        bibtex_lines.append(f'  journal = "{{{journal}}}",')

        if metadata.get("volume"):
            bibtex_lines.append(f'  volume = "{metadata["volume"]}",')
        if metadata.get("issue"):
            bibtex_lines.append(f'  number = "{metadata["issue"]}",')
        if metadata.get("page"):
            bibtex_lines.append(f'  pages = "{metadata["page"]}",')

    elif entry_type in ["incollection", "inproceedings"] and metadata.get(
        "container-title"
    ):
        booktitle = metadata["container-title"][0]
        bibtex_lines.append(f'  booktitle = "{{{booktitle}}}",')

    # Publisher
    if metadata.get("publisher"):
        bibtex_lines.append(f'  publisher = "{{{metadata["publisher"]}}}",')

    # DOI
    if metadata.get("DOI"):
        bibtex_lines.append(f'  doi = "{{{metadata["DOI"]}}}",')

    # URL
    if metadata.get("URL"):
        bibtex_lines.append(f'  url = "{{{metadata["URL"]}}}",')

    bibtex_lines.append("}")

    return "\n".join(bibtex_lines)


def fetch_crossref_metadata(doi: str) -> Optional[Dict]:
    """Fetch metadata from Crossref API."""
    clean_doi_str = clean_doi(doi)
    url = f"{CROSSREF_API_BASE}{clean_doi_str}"

    headers = {"User-Agent": f"ArtContext/1.0 (mailto:{CROSSREF_MAILTO})"}

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {})
        elif response.status_code == 404:
            print(f"  DOI not found: {clean_doi_str}")
        else:
            print(f"  Error {response.status_code} for DOI: {clean_doi_str}")
    except requests.exceptions.RequestException as e:
        print(f"  Request failed for DOI {clean_doi_str}: {e}")

    return None


def enrich_work_metadata(work_id: str, work_data: Dict) -> Tuple[bool, Dict]:
    """Enrich a single work with metadata."""
    doi = work_data.get("DOI")
    if not doi:
        return False, work_data

    # Skip if already enriched
    if all(key in work_data for key in ["Author_Name", "Work_Title", "Year", "BibTeX"]):
        return True, work_data

    # Fetch metadata from Crossref
    metadata = fetch_crossref_metadata(doi)
    if not metadata:
        return False, work_data

    # Extract and format fields
    enriched = work_data.copy()

    # 1. Author name(s)
    if metadata.get("author"):
        enriched["Author_Name"] = format_authors(metadata["author"])
    else:
        enriched["Author_Name"] = "Unknown Author"

    # 2. Work title
    if metadata.get("title"):
        enriched["Work_Title"] = metadata["title"][0]
    else:
        enriched["Work_Title"] = "Unknown Title"

    # 3. Year
    date_parts = metadata.get("published-print", {}).get("date-parts", [[]])
    if not date_parts or not date_parts[0]:
        date_parts = metadata.get("published-online", {}).get("date-parts", [[]])

    if date_parts and date_parts[0]:
        enriched["Year"] = str(date_parts[0][0])
    else:
        enriched["Year"] = "n.d."

    # 4. BibTeX
    enriched["BibTeX"] = generate_bibtex(metadata, work_id)

    return True, enriched


def main():
    """Process all works and enrich with metadata."""
    print("Loading works.json...")
    with open(WORKS_FILE, "r") as f:
        works = json.load(f)

    print(f"Found {len(works)} works to process\n")

    enriched_works = {}
    success_count = 0

    # Process each work
    for work_id, work_data in tqdm(works.items(), desc="Enriching works"):
        success, enriched_data = enrich_work_metadata(work_id, work_data)
        enriched_works[work_id] = enriched_data

        if success:
            success_count += 1

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    # Save enriched data
    print(f"\nWriting enriched data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(enriched_works, f, indent=2)

    print(f"\nComplete! Successfully enriched {success_count}/{len(works)} works")
    print(f"Results saved to: {OUTPUT_FILE}")

    # Show sample of enriched data
    if enriched_works:
        sample_id = list(enriched_works.keys())[0]
        sample = enriched_works[sample_id]
        if "Author_Name" in sample:
            print("\nSample enriched entry:")
            print(f"  Work ID: {sample_id}")
            print(f"  Authors: {sample.get('Author_Name', 'N/A')}")
            print(f"  Title: {sample.get('Work_Title', 'N/A')}")
            print(f"  Year: {sample.get('Year', 'N/A')}")
            print(
                f"  BibTeX preview: {sample.get('BibTeX', 'N/A').split(chr(10))[0]}..."
            )


if __name__ == "__main__":
    main()
