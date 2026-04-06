from pprint import pprint

from src.data_prep import load_and_prepare_dataset
from src.pipeline import run_pipeline

DATA_PATH = "data/USDA_V2_merged.json"


def main():
    print("Loading dataset...")
    dataset = load_and_prepare_dataset(DATA_PATH)
    print(f"Loaded {len(dataset)} usable dataset rows.")

    text = input("\nEnter a meal description: ").strip()
    if not text:
        print("No input provided.")
        return

    result = run_pipeline(
        text=text,
        dataset=dataset,
        save_unmatched_candidates=True,
    )

    print("\n" + "=" * 60)
    print("OPENROUTER PROTOTYPE RESULT")
    print("=" * 60)
    pprint(result)


if __name__ == "__main__":
    main()