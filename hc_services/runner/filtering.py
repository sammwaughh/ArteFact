"""
Filtering logic for sentence selection based on topics and creators.
"""

import json
from pathlib import Path
from typing import List, Set, Dict, Any

# Load data files
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "hc_services" / "runner" / "data"

# Load all necessary data
with open(DATA_DIR / "sentences.json", "r", encoding="utf-8") as f:
    SENTENCES = json.load(f)

with open(DATA_DIR / "works.json", "r", encoding="utf-8") as f:
    WORKS = json.load(f)

with open(DATA_DIR / "topics.json", "r", encoding="utf-8") as f:
    TOPICS = json.load(f)


def get_filtered_sentence_ids(
    filter_topics: List[str] = None,
    filter_creators: List[str] = None
) -> Set[str]:
    """
    Get the set of sentence IDs that match the given filters.
    
    Args:
        filter_topics: List of topic codes to filter by (e.g., ["C2778983918", "C520712124"])
        filter_creators: List of creator names to filter by
        
    Returns:
        Set of sentence IDs that match all filters
    """
    # Start with all sentence IDs
    valid_sentence_ids = set(SENTENCES.keys())
    
    # If no filters, return all sentences
    if not filter_topics and not filter_creators:
        return valid_sentence_ids
    
    # Build set of valid work IDs based on filters
    valid_work_ids = set()
    
    # Apply topic filter
    if filter_topics:
        # Method 1: Using topics.json (topic -> works mapping)
        # For each selected topic, get all works that have it
        for topic_id in filter_topics:
            if topic_id in TOPICS:
                # Add all works that have this topic
                valid_work_ids.update(TOPICS[topic_id])
        
        # Alternative Method 2: Using works.json (work -> topics mapping)
        # This would be less efficient but equivalent:
        # for work_id, work_data in WORKS.items():
        #     work_topics = work_data.get("TopicIDs", [])
        #     if any(topic in work_topics for topic in filter_topics):
        #         valid_work_ids.add(work_id)
    else:
        # If no topic filter, all works are valid so far
        valid_work_ids = set(WORKS.keys())
    
    # Apply creator filter
    if filter_creators:
        # Further filter by creator
        if filter_topics:
            # If we already filtered by topics, only check those works
            creator_filtered_works = set()
            for work_id in valid_work_ids:
                work_data = WORKS.get(work_id, {})
                work_artist = work_data.get("Artist", "")
                if work_artist in filter_creators:
                    creator_filtered_works.add(work_id)
            valid_work_ids = creator_filtered_works
        else:
            # If no topic filter was applied, check all works
            valid_work_ids = set()
            for work_id, work_data in WORKS.items():
                work_artist = work_data.get("Artist", "")
                if work_artist in filter_creators:
                    valid_work_ids.add(work_id)
    
    # Now filter sentences to only those from valid works
    filtered_sentence_ids = set()
    for sentence_id in valid_sentence_ids:
        # Extract work ID from sentence ID (format: WORKID_sXXXX)
        work_id = sentence_id.split("_")[0]
        if work_id in valid_work_ids:
            filtered_sentence_ids.add(sentence_id)
    
    return filtered_sentence_ids


def apply_filters_to_results(
    results: List[Dict[str, Any]],
    filter_topics: List[str] = None,
    filter_creators: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter a list of results based on topics and creators.
    
    Args:
        results: List of result dictionaries with 'sentence_id' field
        filter_topics: List of topic codes to filter by
        filter_creators: List of creator names to filter by
        
    Returns:
        Filtered list of results
    """
    if not filter_topics and not filter_creators:
        return results
    
    valid_sentence_ids = get_filtered_sentence_ids(filter_topics, filter_creators)
    
    # Filter results to only include valid sentences
    filtered_results = [
        result for result in results 
        if result.get("sentence_id") in valid_sentence_ids
    ]
    
    # Re-rank the filtered results
    for i, result in enumerate(filtered_results, 1):
        result["rank"] = i
    
    return filtered_results