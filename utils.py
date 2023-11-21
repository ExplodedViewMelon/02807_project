import pandas as pd
from tqdm import tqdm


def get_referenced_by(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with a column 'referenced_by' that contains a list of IDs that reference the current ID.
    """
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
    full_df["n_references"] = full_df["references"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return full_df

#Filtering
filtered_df=full_df[(full_df["n_counted_citations"]!=0) & (full_df["n_references"]!=0) &
                    (full_df["abstract"]!="") & (full_df["title"]!="")]

#TF-IDF on the user input + jaccard similarity function 
from collections import Counter

def input_tf_idf(user_input):
    tokenize_input=user_input.lower().split()
    word_count=Counter(tokenize_input)
    total_words=len(tokenize_input)
    tf={word:count/total_words for word,count in word_count.items()}    
    return tf


def jaccard_similarity(tf,keywords):
    input_words=list(tf.keys())
    union=0
    intersection=0
    set_input_words=set(input_words)
    set_keywords=set(keywords)
    for keyword in set_input_words:
        if keyword in set_keywords:
            intersection=intersection+1
    unionlist=input_words+keywords
    unionset=set(unionlist)
    union=len(unionset)
    similarity=intersection/union
    return similarity
