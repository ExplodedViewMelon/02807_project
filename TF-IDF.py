# -*- coding: utf-8 -*-
import pandas as pd
import random
from faker import Faker
import networkx as nx
import string
from networkx.algorithms.community.centrality import girvan_newman
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
fake = Faker()
translator = str.maketrans("", "", string.punctuation + string.digits)
stop_words = set(stopwords.words("english"))
porter_stemmer = PorterStemmer()
tfidf_vectorizer = TfidfVectorizer(stop_words=None, tokenizer=None)

def girvan_newman_directed(graph):
    
    G = graph.copy()
    num_components = 0
    
    while G.number_of_edges() > 0:
        #betweenness centrality for all edges,get highest,remove edge
        edge_betweenness = nx.edge_betweenness_centrality(G)
        max_edge = max(edge_betweenness, key=edge_betweenness.get)
        G.remove_edge(*max_edge)

        num_components = nx.number_weakly_connected_components(G)
        if num_components > 1:
            break
    return list(nx.weakly_connected_components(G))

# Sample data
data = {
    'Abstract': [fake.paragraph() for _ in range(1, 11)],
    'Title': [fake.sentence() for _ in range(1, 11)],
    'References': [[] for _ in range(1, 11)],
    'Paper_ID': [i for i in range(1, 11)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some random references
for i in range(1, 11):
    num_references = random.randint(0, 3)  # Random number of references (0 to 3)
    references = random.sample(range(1, 11), num_references)  # Random references from other papers
    df.at[i - 1, 'References'] = references


papers = {}

for index, row in df.iterrows():
    paper_id = row["Paper_ID"]
    title = row["Title"]
    abstract = row["Abstract"]
    paper_references = row["References"]

    if paper_id not in papers:
        papers[paper_id] = {"Title": title, "Abstract": abstract, "References": []}

    # Ensure references are in list format
    papers[paper_id]["References"].extend(paper_references)


graph = nx.DiGraph()

# Add nodes and edges to the graph based on the references
for paper_id, data in papers.items():
    graph.add_node(paper_id)
    for reference_id in data["References"]:
        graph.add_edge(paper_id, reference_id)
     
communities = girvan_newman_directed(graph)
test=girvan_newman(graph)
test_list=list(test)
papers_keywords = {}
for community in communities:
    for paper_id in community:
        title=papers[paper_id]["Title"]
        abstract=papers[paper_id]["Abstract"]
        combined_text = f"{title} {abstract}"
        #lowercase
        combined_text_lower = combined_text.lower()
        #punctuation and numbers 
        combined_text_cleaned = combined_text_lower.translate(translator)
        #stopwords
        words = combined_text_cleaned.split()
        filtered_words = [word for word in words if word not in stop_words]

        # stem
        stemmed_words = [porter_stemmer.stem(word) for word in filtered_words]
        combined_text_final = " ".join(stemmed_words)
        # TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform([combined_text_final])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        # keywords
        top_keywords = sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)
        papers_keywords[paper_id] = {"Keywords": top_keywords}