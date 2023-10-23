import polars as pl
from tqdm import tqdm


class TrainRawIdProcessing:
    def get_collecting(self, collecting_path):
        triplets = {}
        pbar = tqdm(desc="Collecting pairs")
        with open(collecting_path, encoding="utf-8") as f:
            for line in f:
                #query, positive, _ = map(int, line.rstrip().split("\t"))
                query, positive, _ = line.rstrip().split("\t")
                    
                if triplets.get(int(query)) is None:
                    triplets[query] = [positive]
                elif positive not in triplets[query]:
                    triplets[query].append(positive)
                pbar.update(1)
        return triplets

    def get_negatives(self, run_path):
        negatives = {}
        pbar = tqdm(desc="Collecting negatives", leave=False)
        
        with open(run_path, encoding="utf-8") as f:
            for line in f:
                query, negative, _ = map(int, line.rstrip().split("\t"))
                if negatives.get(query) is None:
                    negatives[query] = [negative]
                elif negative not in negatives[query]:
                    negatives[query].append(negative)
                pbar.update(1)
        return negatives
    
    def run(self, run_paths, collecting_paths, save_paths, min_num_negatives:int = 4):
        query_pairs = {}
        for run_path, collecting_path, save_path in tqdm(zip(run_paths, collecting_paths, save_paths), total=len(run_paths), desc="Processing runs"):
            pairs = self.get_collecting(collecting_path) # get collecting
            
            negatives = self.get_negatives(run_path)
            for query, positives in tqdm(pairs.items(), total=len(pairs), desc="Generate query-pairs"):
                if negatives.get(query) is not None:
                    query_pairs[query] = {
                        "positives": positives,
                        "negatives": negatives[query],
                    }
                    
            df = pl.DataFrame({
                "query": [query for query in query_pairs.keys()],
                "positives": [pair["positive"] for pair in query_pairs.values()],
                "negatives": [pair["negative"] for pair in query_pairs.values()],
            })
            
            df = df.explode("positive")
            df = df.filter(pl.col("negatives").arr.lengths() >= min_num_negatives)
            df.write_ndjson(save_path)

if __name__ == '__main__':
    collecting_paths = [r"data\filtered_english_collection.tsv", r"data\filtered_vietnamese_collection.tsv"]
    run_paths = [r"data\runs\run.bm25_english-msmarco.txt", r"data\runs\run.bm25_vietnamese-msmarco.txt"]
    save_paths = [r"data\interim\english_train.ids.json", r"data\interim\vietnamese_train.ids.json"]
    
    raw_prossesor = TrainRawIdProcessing()
    for idx in range(len(run_paths)):
        raw_prossesor.run(run_paths=run_paths,
                        collecting_paths=collecting_paths,
                        save_paths=save_paths
                        )
