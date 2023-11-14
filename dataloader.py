import pandas as pd
import os
import requests
from tqdm import tqdm
import requests
import zipfile
import networkx as nx


class CitationDataset:
    """
    Object for downloading, unzipping and loading the citation dataset.
    If you already have the downloaded zip file, put it into a folder named DATA.
    """

    def __init__(self, cache_dir: str = "") -> None:
        if not cache_dir:
            self._base_path = os.getcwd()
        else:
            self._base_path = cache_dir

        self._folder_path: str = self._base_path + "/DATA"
        self._subfolder_path: str = self._folder_path + "/dblp-ref"
        self._zip_path: str = self._folder_path + "/dblp.v10.zip"
        self._file_paths: list[str] = [
            self._subfolder_path + f"/dblp-ref-{i}.json" for i in range(4)
        ]
        self._file_url = f"https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip"

        if not self._cache_exists():
            if not self._zip_exists():
                self._download_to_cache()
            self._unzip()

    def _zip_exists(self) -> bool:
        return os.path.isfile(self._zip_path)

    def _cache_exists(self) -> bool:
        """check that all four files exists in the folder"""
        return all((os.path.isfile(file_path) for file_path in self._file_paths))

    def _download_to_cache(self) -> None:
        print(f"Downloading data from {self._file_url}")

        if not os.path.exists(self._folder_path):
            os.makedirs(self._folder_path)

        # download content with context bar
        response = requests.get(self._file_url, stream=True)
        progress_bar = tqdm(
            total=int(response.headers.get("content-length", 0)),
            unit="B",
            unit_scale=True,
        )

        with open(self._zip_path, "wb") as file:
            for data in response.iter_content(chunk_size=4096):
                file.write(data)
                progress_bar.update(len(data))  # Update the progress bar

        progress_bar.close()

    def _unzip(self) -> None:
        print("unzipping", self._zip_path)
        with zipfile.ZipFile(self._zip_path, "r") as zip_ref:
            zip_ref.extractall(self._folder_path)

    def load_dataframe(self, *, subset=False) -> pd.DataFrame:
        """
        Load data from cache as a pandas DataFrame.
        Set subset = True to only use the smallest of the four files.
        """
        print("loading dataframe from cache", self._subfolder_path)

        df = pd.DataFrame()
        for file in reversed(self._file_paths):
            print("loading", file)
            df = pd.concat((df, pd.read_json(file, lines=True)))
            if subset:  # stop after reading the last, and smallest of the json files
                break

        df.references = df.references.fillna("")
        df.abstract = df.abstract.fillna("")

        return df

    def load_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Loads networkx DiGraph from dataframe.
        """
        # Create an empty directed graph
        G = nx.DiGraph()
        
        # Iterate through the DataFrame and add nodes and edges to the graph
        for _, row in tqdm(df.iterrows(), total=len(df)):
            node_id = row.id
            references = row.references

            # Add the node to the graph
            G.add_node(
                node_id,
                abstract=row.abstract,
                authors=row.authors,
                n_citation=row.n_citation,
                title=row.title,
                venue=row.venue,
                year=row.year,
            )

            # Add edges from the node to its references
            for reference in references:
                if reference in df['id']:
                    G.add_edge(node_id, reference)
                else:
                    print(f"Reference {reference} not found in the DataFrame.")
        return G


if __name__ == "__main__":
    dataset = CitationDataset()
