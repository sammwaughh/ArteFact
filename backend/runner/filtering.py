"""
Filtering logic for sentence selection based on topics and creators.
"""

from typing import Any, Dict, List, Set

# Import data from config (loaded from HF datasets)
from .config import sentences, works, creators, topics, topic_names

# Data is now loaded from Hugging Face datasets in config.py
# No need to load from local files anymore

def get_filtered_sentence_ids(
    filter_topics: List[str] = None, filter_creators: List[str] = None
) -> Set[str]:
    """
    Get the set of sentence IDs that match the given filters.
    """
    # Start with all sentence IDs from the sentences dictionary
    valid_sentence_ids = set(sentences.keys())

    # If no filters, return all sentences
    if not filter_topics and not filter_creators:
        return valid_sentence_ids

    # Build set of valid work IDs based on filters
    valid_work_ids = set()

    # Apply topic filter
    if filter_topics:
        # Using topics dictionary (topic -> works mapping)
        # For each selected topic, get all works that have it
        for topic_id in filter_topics:
            if topic_id in topics:
                # Add all works that have this topic
                valid_work_ids.update(topics[topic_id])
    else:
        # If no topic filter, all works are valid so far
        valid_work_ids = set(works.keys())

    # Apply creator filter
    if filter_creators:
        # Direct lookup in creators dictionary (more efficient)
        creator_work_ids = set()
        for creator_name in filter_creators:
            if creator_name in creators:
                # Get all works by this creator directly from creators dictionary
                creator_work_ids.update(creators[creator_name])

        # Intersect with existing valid_work_ids if topics were filtered
        if filter_topics:
            valid_work_ids = valid_work_ids.intersection(creator_work_ids)
        else:
            valid_work_ids = creator_work_ids

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
    filter_creators: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter a list of results based on topics and creators.

    Args:
        results: List of result dictionaries with 'id' field
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
        result for result in results if result.get("id") in valid_sentence_ids
    ]

    # Re-rank the filtered results
    for i, result in enumerate(filtered_results, 1):
        result["rank"] = i

    return filtered_results
