import json

path = "data/sample_data.json"

def load_preference_data(path):
    with open(path, "r") as f:
        return json.load(f)
import json
