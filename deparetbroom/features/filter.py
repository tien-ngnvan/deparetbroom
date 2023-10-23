import pandas as pd
import polars as pl
from tqdm import tqdm


collection_path = "mmarco/raw/collections/{language}_collection.tsv"
filtered_collection_path = "mmarco/raw/collections/filtered_{language}_collection.tsv"

queries_path = "mmarco/raw/queries/{language}_queries.train.tsv"
filtered_queries_path = "mmarco/raw/queries/filtered_{language}_queries.train.tsv"

train_path = "mmarco/raw/triples.train.ids.small.tsv"
filtered_train_path = "mmarco/raw/filtered.triples.train.ids.small.tsv"

languages = ["vietnamese", "english"]


for language in tqdm(languages, total=len(languages)):
    collection = pl.read_csv(
        collection_path.format(language=language),
        separator="\t",
        new_columns=["doc_id", "doc"]
    )

    try:
        queries = pl.read_csv(
            queries_path.format(language=language),
            separator="\t",
            new_columns=["query_id", "query"]
        )
    except:
        queries = pl.from_pandas(
            pd.read_csv(
                queries_path.format(language=language),
                delimiter="\t",
                header=0,
                names=["query_id", "query"]
            )
        )

    train = pl.read_csv(
        train_path.format(language=language),
        separator="\t",
        new_columns=["query_id", "q_score", "positive", "p_score"]
    )

    collection = collection.with_columns(
        pl.col("doc").map_elements(lambda x: len(x.split())).alias("doc_len")
    )
    collection = collection.filter((pl.col("doc_len") < 1000) & (pl.col("doc_len") > 1))
    doc_ids = collection.get_column("doc_id").to_list()
    train = train.filter(
        (pl.col("pos_id").is_in(doc_ids)) & (pl.col("neg_id").is_in(doc_ids))
    )
    query_ids = train.get_column("query_id").to_list()
    queries = queries.filter(pl.col("query_id").is_in(query_ids))

    queries.write_csv(
        filtered_queries_path.format(language=language),
        separator="\t",
        has_header=False
    )

    train.write_csv(
        filtered_train_path,
        separator="\t",
        has_header=False
    )

    collection.write_csv(
        filtered_collection_path.format(language=language),
        separator="\t",
        has_header=False
    )