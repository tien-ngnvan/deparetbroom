import polars as pl
from tqdm import tqdm


def generate_query_pairs(qp_df, qn_df):
    query_pairs = {}
    for query in tqdm(qp_df["query"].unique(), total=qp_df["query"].n_unique(), desc="Generate query-pairs", leave=False):
        positives = qp_df.filter(pl.col("query") == query)["positive"].to_list()
        negatives = qn_df.filter(pl.col("query") == query)["negative"].to_list()
        if len(negatives) < min_num_negatives:
            continue
        query_pairs[query] = {
            "positive": positives,
            "negative": negatives,
        }

    return query_pairs


if __name__ == '__main__':
    
    dev_path = "mmarco/raw/qrels.dev.tsv"
    run_paths = ["mmarco/raw/runs/run.bm25_english-msmarco.txt", "mmarco/raw/runs/run.bm25_vietnamese-msmarco.txt"]
    save_paths = ["mmarco/interim/english_dev.ids.json", "mmarco/interim/vietnamese_dev.ids.json"]
    min_num_negatives = 4 # samples with less than negatives is remove


    # read query - positive
    qp_df = pl.read_csv(dev_path, separator="\t", new_columns=["query", "q_score", "positive", "p_score"])[["query", "positive"]]

    for run_path, save_path in tqdm(zip(run_paths, save_paths), total=len(run_paths), desc="Processing runs"):
        qn_df = pl.read_csv(run_path, separator="\t", new_columns=["query", "negative", "score"])[["query", "negative"]]
        query_pairs = generate_query_pairs(qp_df, qn_df)
        df = pl.DataFrame({
            "query": [query for query in query_pairs.keys()],
            "positive": [pair["positive"] for pair in query_pairs.values()],
            "negatives": [pair["negative"] for pair in query_pairs.values()],
        })

        print('-' * 50 + ' ' + save_path.split('_')[0] + ' ' + '-' * 50)
        print(df.head(10))
        df = df.explode("positive")
        df.write_ndjson(save_path)