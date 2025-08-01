# test_api_heatmap.py
import base64
from pathlib import Path

import requests

# Use one of your existing run IDs
run_id = "ce00a2134d09467bb2f60ceca5ce4472"  # Replace with your actual run ID
sentence = "The fingers almost touch in the centre of the painting"

# Call the API
response = requests.post(
    "http://localhost:8000/heatmap",
    json={"runId": run_id, "sentence": sentence, "layerIdx": -1},
)

if response.status_code == 200:
    data = response.json()
    data_url = data["dataUrl"]

    # Extract base64 data
    _, base64_data = data_url.split(",", 1)

    # Decode and save
    image_data = base64.b64decode(base64_data)
    Path("test-outputs/api_heatmap_test.png").write_bytes(image_data)
    print("✓ Heatmap saved to test-outputs/api_heatmap_test.png")
else:
    print(f"✗ Error: {response.status_code}")
    print(response.json())
