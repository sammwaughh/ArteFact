import json
import time

import requests

out = []
cursor = "*"
while cursor:
    r = requests.get(
        "https://api.openalex.org/concepts",
        params={
            "filter": "ancestors.id:C52119013",
            "per_page": 200,
            "cursor": cursor,
        },
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    out.extend([c["id"].rsplit("/", 1)[-1] for c in data["results"]])
    cursor = data["meta"]["next_cursor"]
    time.sleep(0.2)  # polite
print(json.dumps(out, indent=2))
