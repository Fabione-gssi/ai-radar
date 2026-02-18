from pathlib import Path
from radar.pipeline import enrich_and_store

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    stats = enrich_and_store(base_dir)
    print("Pipeline completata:")
    for k, v in stats.items():
        print(f"- {k}: {v}")
