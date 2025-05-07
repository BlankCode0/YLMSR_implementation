from utils import load_preference_data

if __name__ == "__main__":
    data = load_preference_data("data/sample_data.json")
    print(f"Loaded {len(data)} samples")
    print(data[0])
from utils import load_preference_data
