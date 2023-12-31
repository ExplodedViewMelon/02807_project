import mmh3
import sys
import os
import pickle
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import get_referenced_by, filter_df, clean_text
from dataloader import CitationDataset
from pandarallel import pandarallel


def shingle(text: str, shingle_size):
        text_list = text.split()
        return list(set(" ".join(text_list[i:i+shingle_size]) for i in range(len(text_list)-shingle_size+1)))
    
def minhash(text_list, seed) -> int:
    hash_list = [mmh3.hash(shingle, seed) for shingle in text_list] 
    return min(hash_list)

def get_signature(text: str, shingle_size = 3, sig_len = 50):    
    shingle_list = shingle(text, shingle_size)
    if len(shingle_list) == 0:
        return np.nan
    try:
        signature = [minhash(shingle_list, seed) for seed in range(sig_len)]
    except  Exception as e:
        print(text)
        print(shingle_list)
        sys.exit(e)
    return signature


def get_model(save_path: str, shingle_size=2, signature_size=250, subset=False) -> pd.DataFrame:
    if os.path.exists(save_path):
        print("Loading cached dataframe...")
        with open(save_path, "rb") as f:
            df = pickle.load(f)
    else:
        print("Loading dataframe...")
        dataset = CitationDataset()
        df = dataset.load_dataframe(subset=subset)
        
        df = get_referenced_by(df)
        print("Filtering dataframe...")
        df = filter_df(df)

        print("Calculating signatures...")
        df["signature"] = df["abstract"].parallel_apply(get_signature, 
                                                        shingle_size=shingle_size, 
                                                        sig_len=signature_size)
        with open(save_path, "wb") as f:
            pickle.dump(df, f)
    return df


def jaccard2(signature1, signature2):
    import numpy as np

    if signature1 == np.nan or signature2 == np.nan:
        return 0

    signatures_doc1 = np.array(signature1)
    signatures_doc2 = np.array(signature2)
    return len(np.intersect1d(signatures_doc1, signatures_doc2)) / len(
        np.union1d(signatures_doc1, signatures_doc2)
    )


def get_most_similar(df, promt, shingle_size=2, signature_size=250, n_top=10):
    clean_promt = clean_text(promt)
    promt_sig = get_signature(clean_promt, 
                              shingle_size=shingle_size, 
                              sig_len=signature_size)

    df["sim"] = df["signature"].parallel_apply(jaccard2, signature2=promt_sig)
    
    return df.nlargest(n_top, ["sim", "n_counted_citations"])


if __name__ == "__main__":
    
    pandarallel.initialize(nb_workers=10, progress_bar=True)
    
    shingle_size = 2
    signature_length = 250
    subset = False
    prompt = None
    
    model_name = f"model_df_{shingle_size}_{signature_length}{'_subset' if subset else ''}.pkl"
    
    model_df = get_model(model_name, 
                         shingle_size=shingle_size, 
                         signature_size=signature_length, 
                         subset=subset)
    
    if len(sys.argv) == 2:
        try:
            filename = sys.argv[1]
            output_file = filename + ".search_result.json"
            with open(sys.argv[1], "r") as f:
                prompt = f.read()
                
        except FileNotFoundError as e:
            sys.exit(f"File {sys.argv[1]} not found.")
    
    if not prompt:
        print("No prompt given, using random abstract from dataset.")
        test_idx = 500
        output_file = "test.search_result.json"
        prompt = model_df.iloc[test_idx]["abstract"]
        model_df = model_df.drop([test_idx], axis=0).reset_index(drop=True)
    
    print()
    print(f"{prompt=}")

    print("Finding most similar papers...")
    df_top = get_most_similar(model_df, 
                              prompt,
                              shingle_size=shingle_size,
                              signature_size=signature_length, 
                              n_top=10)
    print()
    
    print(f"Writing results to {output_file}...")
    with open(output_file, "w") as f:
        f.write(json.dumps(json.loads(df_top[['title', 'abstract', 'authors', 'sim', 'n_references', 'n_counted_citations']].to_json(orient='records')), indent=4))
        f.write("\n\n")
