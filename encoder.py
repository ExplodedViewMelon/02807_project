from typing import List
from dataloader import CitationDataset
import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm


class Search:
    def __init__(
        self,
        path_data: str = "/work3/s174032/02807_project_data",
        make_embeddings=False,
        make_index=False,
    ):
        self.path_data = path_data

        self.dataset = CitationDataset(cache_dir=self.path_data)

        if make_embeddings:
            self.make_and_save_embeddings()
        print("loading embeddings")
        self.df = self.load_embeddings()

        if make_index:
            self.make_and_save_faiss_index()
        print("loading index")
        self.index = self.load_index()

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
        )

        self.G = self.get_graph()

    def get_graph(self) -> nx.DiGraph:
        return self.dataset.load_graph(self.df)

    def make_and_save_embeddings(self) -> None:
        """
        loads data using dataloader, encodes and saves dataframe as pickle.
        """

        # load dataset

        df = self.dataset.load_dataframe()

        df["content"] = df.title + ". " + df.abstract

        # remove empty abstracts
        df = df.query("content != '' and content != '. '")

        # get model from hugging face
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
        )

        # calculate embeddings
        df["embeddings"] = list(
            model.encode(df.abstract.to_list(), show_progress_bar=True)
        )

        pd.to_pickle(df, self.path_data + "/df_with_encodings.pkl")

    def load_embeddings(self) -> pd.DataFrame:
        return pd.read_pickle(self.path_data + "/df_with_encodings.pkl")

    def make_and_save_faiss_index(self, batch_size=1000) -> None:
        vector_dim = 384
        # Create an index
        index = faiss.IndexFlatL2(vector_dim)  # L2 distance index

        # Add vectors to the index
        vectors = self.df.embeddings.to_list()
        for i in tqdm(range(0, len(vectors), batch_size), desc="Batch"):
            index.add(np.array(vectors[i : i + batch_size]))

        print("writing index")
        faiss.write_index(index, self.path_data + "/encodings.index")

    def load_index(self) -> faiss.Index:
        index = faiss.read_index(self.path_data + "/encodings.index")
        return index

    def search_index(self, search_sentences, index, k=5):
        # return distances, k nearest neighbor indexes

        query_vector = self.model.encode(search_sentences)

        # Search in the index
        return index.search(query_vector, k)

    def search_index_vocal(self, query: str, top_k=5) -> None:
        # TODO: rewrite to handle multiple queries
        distances, neighbors = self.search_index([query], self.index, k=top_k)

        # Print the results
        print("\nNearest Neighbors:")
        for index, row in self.df.iloc[list(neighbors[0])].iterrows():
            print(index)
            print(row.title)
            print(row.abstract.strip())

        print("\nDistances:")
        print(distances)


if __name__ == "__main__":
    """will recreate both embeddings and index. Takes ~2 hours."""
    S = Search(make_embeddings=False, make_index=True)

# usecase:
# S = Search()
# S.search_index_vocal(["Some prompts"])
