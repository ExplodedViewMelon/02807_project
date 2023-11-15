import pandas as pd
from tqdm import tqdm
from typing import List


def get_referenced_by(df: pd.DataFrame) -> List[int]:
    ref_df = df[["id", "references"]]
    reversed_refs = {}
    for _, row in tqdm(ref_df.iterrows(), total=ref_df.shape[0]):
        for ref in row["references"]:
            # Add the current ID to the list of IDs that reference 'ref'
            if ref in reversed_refs:
                reversed_refs[ref].append(row["id"])
            else:
                reversed_refs[ref] = [row["id"]]

    reversed_df = pd.DataFrame(
        list(reversed_refs.items()), columns=["id", "referenced_by"]
    )

    full_df = pd.merge(df, reversed_df, on="id", how="outer")
    full_df["n_counted_citations"] = full_df["referenced_by"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    full_df["n_citations"] = full_df["references"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return full_df
