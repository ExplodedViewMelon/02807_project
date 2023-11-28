from typing import List
from dataloader import CitationDataset
import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import itertools
import time


class Search:
    def __init__(
        self,
        path_data: str = "/work3/s174032/02807_project_data",
        make_embeddings=False,
        make_index=False,
        use_subset=False,
    ):
        self.path_data = path_data
        self.use_subset = use_subset

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

        df = self.dataset.load_dataframe(subset=self.use_subset)

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
            index.add(np.array(vectors[i : i + batch_size]))  # type: ignore

        print("writing index")
        faiss.write_index(index, self.path_data + "/encodings.index")

    def load_index(self) -> faiss.Index:
        index = faiss.read_index(self.path_data + "/encodings.index")
        return index

    def search_index(self, search_sentences, k=5):
        # return distances, k nearest neighbor indexes

        query_vector = self.model.encode(search_sentences)

        # Search in the index
        return self.index.search(query_vector, k)

    def search_index_vocal(self, query: str, top_k=5) -> None:
        distances, neighbors = self.search_index([query], k=top_k)

        # Print the results
        print("\nNearest Neighbors:")
        for index, row in self.df.iloc[list(neighbors[0])].iterrows():
            print(index)
            print(row.title)
            print(row.abstract.strip())

        print("\nDistances:")
        print(distances)

    def find_similar_papers(
        self, id: str = "001eef4f-1d00-4ae6-8b4f-7e66344bbc6e", k: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        returns tuple of
        [related_articles_not_referenced, related_articles_referenced]
        """

        def flatten(nested_list):
            return list(itertools.chain.from_iterable([nested_list]))

        # get paper from id
        row = self.df.query("id == @id").iloc[0]
        rows_references = row.references

        # get papers nearest encodings
        distances, neighbors = self.index.search((row.embeddings.reshape(1, -1)), k)  # type: ignore
        rows_vector_search = self.df.iloc[list(neighbors[0])]

        rows_non_overlapping = rows_vector_search[
            ~rows_vector_search.id.isin(rows_references)
        ]
        rows_overlapping = rows_vector_search[
            rows_vector_search.id.isin(rows_references)
        ]

        return rows_non_overlapping, rows_overlapping


# Random search prompt for benchmarking the search.
search_prompts = [
    "The Impact of Climate Change on Arctic Biodiversity",
    "Advances in Quantum Computing and Its Applications",
    "CRISPR-Cas9 and Gene Editing: Ethical Implications",
    "Machine Learning Algorithms in Predictive Medicine",
    "The Role of Microbiomes in Human Health",
    "Sustainable Energy Solutions: Solar Power Innovations",
    "Neuroplasticity and Cognitive Rehabilitation Post-Stroke",
    "Artificial Intelligence in Autonomous Vehicle Technology",
    "Dark Matter and Dark Energy: Unraveling Cosmic Mysteries",
    "Nanotechnology in Drug Delivery Systems",
    "The Evolution of Antibiotic Resistance",
    "Bioprinting and the Future of Organ Transplantation",
    "Deep Learning Techniques in Financial Forecasting",
    "The Psychology of Social Media Addiction",
    "Climate Engineering: A Solution or a Threat?",
    "Blockchain Technology in Secure Voting Systems",
    "Astrophysics: Exploring Exoplanet Atmospheres",
    "The Ethics of Human Cloning",
    "Augmented Reality in Education",
    "Understanding the Human Microbiome and Disease",
    "Robotics in Healthcare: Opportunities and Challenges",
    "The Physics of Black Holes",
    "Genetic Basis of Neurodegenerative Diseases",
    "Ocean Acidification and Marine Ecosystems",
    "Virtual Reality Therapy for PTSD",
    "Cybersecurity in the Era of IoT",
    "Molecular Mechanisms of Aging",
    "Renewable Energy: Wind Turbine Efficiency",
    "AI and Bias in Facial Recognition Technology",
    "The Role of Stem Cells in Regenerative Medicine",
    "The Impact of Deforestation on Global Ecosystems",
    "Particle Physics: The Search for the Higgs Boson",
    "The Psychology of Leadership and Management",
    "3D Printing and Its Industrial Applications",
    "The Future of Food: Lab-Grown Meat",
    "Wearable Technology in Personal Health Monitoring",
    "The Science of Sleep and Dreams",
    "Big Data Analytics in Public Health",
    "Coral Reef Restoration Techniques",
    "The Impact of Space Travel on Human Physiology",
    "Quantum Cryptography and Information Security",
    "The Social Implications of AI",
    "Smart Cities: Urban Planning and Sustainability",
    "The Genetics of Cancer",
    "Renewable Energy Storage Solutions",
    "The Effects of Microplastics on Marine Life",
    "Gravitational Waves and Their Detection",
    "Augmented Reality in Surgical Procedures",
    "Climate Change and Agricultural Practices",
    "The Neurobiology of Addiction",
    "Fusion Energy: Progress and Challenges",
    "The Future of AI in Education",
    "Oceanography: Sea Level Rise Predictions",
    "Biometric Authentication Technologies",
    "The Sociology of Urbanization",
    "Advanced Materials in Electronics",
    "Understanding Autism Spectrum Disorders",
    "Telemedicine and Remote Healthcare",
    "Smart Grid Technology and Energy Management",
    "The Impact of Virtual Reality on Entertainment",
    "Environmental Toxicology and Pollution",
    "The Science of Happiness and Well-being",
    "Nanorobots in Medical Diagnostics",
    "The Evolutionary Biology of Extinct Species",
    "Exoplanetary Atmospheres and Habitability",
    "The Role of Artificial Intelligence in Art",
    "Genetic Engineering in Agriculture",
    "Brain-Computer Interfaces",
    "The Physics of Superconductors",
    "Climate Change and Vector-Borne Diseases",
    "Autonomous Drones in Disaster Response",
    "The Psychology of Consumer Behavior",
    "Advanced Algorithms in Cryptography",
    "Social Robotics and Human Interaction",
    "The Impact of 5G Technology",
    "Bioremediation of Polluted Environments",
    "The Science Behind Emotional Intelligence",
    "Virtual Reality in Architectural Design",
    "Quantum Mechanics and Its Philosophical Implications",
    "The Role of Nanotechnology in Energy Conversion",
    "The Biology of Aging and Longevity",
    "Machine Learning in Climate Change Prediction",
    "The Economics of Renewable Energy",
    "The Role of Genetics in Mental Health",
    "Atmospheric Science: Studying Climate Change",
    "Robotics in Precision Agriculture",
    "The Psychology of Online Learning",
    "Materials Science: Developing Stronger Alloys",
    "The Future of Biodegradable Plastics",
    "The Human Brain: Understanding Neural Networks",
    "Astrophotography and Cosmic Phenomena",
    "The Ethics of AI in Healthcare",
    "Urban Ecology and Biodiversity",
    "The Role of Technology in Modern Warfare",
    "Paleoclimatology: Reconstructing Past Climates",
    "The Psychology of Group Dynamics",
    "The Science of Renewable Energy Sources",
    "Advanced Techniques in Forensic Science",
    "The Sociology of Social Networks",
    "The Science of Meditation and Mindfulness",
]


def benchmark():
    S = Search(make_embeddings=False, make_index=False)
    # not_referenced, referenced = S.find_similar_papers()
    # print("similar papers", not_referenced)

    t0 = time.time()
    for search_prompt in search_prompts:
        S.search_index([search_prompt], 5)

    print(time.time() - t0, "s elapsed during 100 searches (top_k=5)")


if __name__ == "__main__":
    """will load both embeddings and index. Setting make_embeddings=True and make_index=True takes ~3 hours using GPU."""
    print("running main script")
    S = Search(make_embeddings=False, make_index=False)
    not_referenced, referenced = S.find_similar_papers()
    print("similar papers", not_referenced)

    id: str = "001eef4f-1d00-4ae6-8b4f-7e66344bbc6e"
    k: int = 10
    # get paper from id
    row = S.df.query("id == @id")
    rows_references = row.references[0]

    # get papers nearest encodings
    distances, neighbors = S.index.search((row.embeddings[0].reshape(1, -1)), k)  # type: ignore
    rows_vector_search = S.df.iloc[list(neighbors[0])]

    rows_non_overlapping = rows_vector_search[
        ~rows_vector_search.id.isin(rows_references)
    ]
    rows_overlapping = rows_vector_search[rows_vector_search.id.isin(rows_references)]

    print(rows_non_overlapping, rows_overlapping)

    for index, r in rows_non_overlapping.iterrows():
        print(r.title)
        print(r.abstract)
        print("-")
