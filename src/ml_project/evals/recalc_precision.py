import json
import statistics
import argparse
from pathlib import Path

def recalc_precision(json_path: Path):
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return
        
    data = json.loads(json_path.read_text(encoding="utf-8"))
    
    for k_key, k_data in data.items():
        queries = k_data.get("per_query", [])
        
        # Отфильтровываем плохие запросы (те, которые упали с ошибкой API)
        valid_queries = []
        for q in queries:
            has_error = any("__error__" in pref for pref in q.get("unmapped_preferences", []))
            if not has_error:
                valid_queries.append(q)
                
        if not valid_queries:
            print(f"{k_key}: No valid queries left.")
            continue
            
        precisions = [q["precision"] for q in valid_queries]
        hits = [1 for q in valid_queries if q["hit"]]
        
        k_data["n_queries"] = len(valid_queries)
        k_data["mean_precision"] = round(statistics.fmean(precisions), 4)
        k_data["median_precision"] = round(statistics.median(precisions), 4)
        k_data["hit_rate"] = round(len(hits) / len(valid_queries), 4)
        k_data["per_query"] = valid_queries
        
        print(f"{k_key}: Re-calculated over {len(valid_queries)} valid queries (removed {len(queries) - len(valid_queries)} bad ones).")
        print(f"  mean_precision = {k_data['mean_precision']}")
        print(f"  median_precision = {k_data['median_precision']}")
        print(f"  hit_rate = {k_data['hit_rate']}")
        
    # Сохраняем обновленный файл
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved updated metrics back to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="metrics/recommender_precision.json")
    args = parser.parse_args()
    recalc_precision(Path(args.input))
