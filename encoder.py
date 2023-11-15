from dataloader import CitationDataset
import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

path_data = "/work3/s174032/02807_project_data"


def encode_and_save_pickle():
    """
    loads data, encodes and saves dataframe as pickle.
    """

    # load dataset
    dataset = CitationDataset(cache_dir=path_data)
    df = dataset.load_dataframe(subset=True)

    # remove empty abstracts
    df = df.query("abstract != ''")

    # get model from hugging face
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    # calculate embeddings
    df["embeddings"] = list(model.encode(df.abstract.to_list(), show_progress_bar=True))

    pd.to_pickle(df, path_data + "/df_with_encodings.pkl")


def make_faiss_index(df):
    vector_dim = 384
    # Create an index
    index = faiss.IndexFlatL2(vector_dim)  # L2 distance index

    # Add vectors to the index
    index.add(np.array(df.embeddings.to_list()))

    return index


def search_index(search_sentences, index, k=5):
    # return distances, k nearest neighbor indexes
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    query_vector = model.encode(search_sentences)

    # Search in the index
    return index.search(query_vector, k)


# encode_and_save_pickle() # run only once.

df = pd.read_pickle("/work3/s174032/02807_project_data/df_with_encodings.pkl")
index = make_faiss_index(df)
distances, neighbors = search_index(["a paper concerning cats."], index)

# Print the results
print("\nNearest Neighbors:")
for index, row in df.iloc[list(neighbors[0])].iterrows():
    print(index)
    print(row.abstract.strip())

print("\nDistances:")
print(distances)
