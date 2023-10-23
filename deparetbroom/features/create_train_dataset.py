import polars as pl
from tqdm import tqdm


def generate_query_pairs(train_path):
    query_pairs = {}
    pbar = tqdm()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            query, positive, negative = map(int, line.rstrip().split("\t"))
            if query_pairs.get(query) is None:
                query_pairs[query] = {
                    "positive": [positive],
                    "negative": [negative],
                }
            else:
                if positive not in query_pairs[query]["positive"]:
                    query_pairs[query]["positive"].append(positive)
                query_pairs[query]["negative"].append(negative)
            pbar.update(1)
    return query_pairs


if __name__ == '__main__':    
    train_path = "mmarco/raw/filtered_triples.train.ids.small.tsv"
    run_paths = ["mmarco/raw/runs/run.bm25_english-msmarco.txt", "mmarco/raw/runs/run.bm25_vietnamese-msmarco.txt"]
    save_paths = ["mmarco/interim/english_train.ids.json", "mmarco/interim/vietnamese_train.ids.json"]

    for run_path, save_path in tqdm(zip(run_paths, save_paths), total=len(run_paths), desc="Processing runs"):
        query_pairs = generate_query_pairs(train_path)
        df = pl.DataFrame({
            "query": [query for query in query_pairs.keys()],
            "positive": [pair["positive"] for pair in query_pairs.values()],
            "negatives": [pair["negative"] for pair in query_pairs.values()],
        })
        df = df.explode("positive")
        df.write_ndjson(save_path)

        print(df.head(10))