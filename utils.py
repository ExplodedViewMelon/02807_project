import pandas as pd
from tqdm import tqdm
from collections import Counter


def clean_text(aString):
    output = aString.replace("\n", "")
    output_list = output.split()
    output_list = ["".join(ch for ch in aWord if ch.isalnum()) for aWord in output_list]
    output_list = [s.lower() for s in output_list]
    output = " ".join(output_list)
    return " ".join(output.split())


def get_referenced_by(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with a column 'referenced_by' that contains a list of IDs that reference the current ID.
    """
    ref_df = df[["id", "references"]]
    reversed_refs = {}
    for _, row in tqdm(
        ref_df.iterrows(), total=ref_df.shape[0], desc="Reversing references"
    ):
        for ref in row["references"]:
            # Add the current ID to the list of IDs that reference 'ref'
            if ref in reversed_refs:
                reversed_refs[ref].append(row["id"])
            else:
                reversed_refs[ref] = [row["id"]]

    reversed_df = pd.DataFrame(
        list(reversed_refs.items()), columns=["id", "referenced_by"]
    )

    full_df = pd.merge(df, reversed_df, on="id", how="left")
    full_df["n_counted_citations"] = full_df["referenced_by"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    full_df["n_references"] = full_df["references"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return full_df


def input_tf_idf(user_input):
    # TF-IDF on the user input + jaccard similarity function
    tokenize_input = user_input.lower().split()
    word_count = Counter(tokenize_input)
    total_words = len(tokenize_input)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf


def jaccard_similarity(tf, keywords):
    input_words = list(tf.keys())
    union = 0
    intersection = 0
    set_input_words = set(input_words)
    set_keywords = set(keywords)
    for keyword in set_input_words:
        if keyword in set_keywords:
            intersection = intersection + 1
    unionlist = input_words + keywords
    unionset = set(unionlist)
    union = len(unionset)
    similarity = intersection / union
    return similarity


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df["abstract"] = df["abstract"].apply(clean_text)
    df = df[df["abstract"] != ""]
    df = df[df["title"] != ""]
    df = df[(df["n_references"] > 0) | (df["n_counted_citations"] > 0)]

    return df.reset_index(drop=True)
