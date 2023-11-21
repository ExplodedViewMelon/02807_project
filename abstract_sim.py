import mmh3
import sys
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import get_referenced_by, filter_df
from dataloader import CitationDataset
from pandarallel import pandarallel


pandarallel.initialize(nb_workers=10, progress_bar=True)


def shingle(text: str, shingle_size):
        text_list = text.split()
        return list(set(" ".join(text_list[i:i+shingle_size]) for i in range(len(text_list)-shingle_size+1)))
    
def minhash(text_list, seed) -> int:
    hash_list = [mmh3.hash(shingle, seed) for shingle in text_list] 
    return min(hash_list)

def get_signature(text: str, shingle_size = 3, sig_len = 5):    
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



def get_model(save_path: str) -> pd.DataFrame:
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            df = pickle.load(f)
    else:
        dataset = CitationDataset()
        df = dataset.load_dataframe(subset=True)

        df = get_referenced_by(df)
        df = filter_df(df)

        k=250
        shingle_size=3
        df["signature"] = df["abstract"].parallel_apply(get_signature, 
                                                        shingle_size=shingle_size, 
                                                        sig_len=k)
        with open(save_path, "wb") as f:
            pickle.dump(df, f)
    return df.convert_dtypes(dtype_backend="pyarrow")




