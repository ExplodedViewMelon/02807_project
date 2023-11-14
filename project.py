# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:38:34 2023

@author: dimsl
"""

import pandas as pd
import networkx as nx


papers_df=pd.read_json(r"C:\Users\dimsl\Desktop\dblp-ref-0.json",lines=True)

for row in papers_df.loc[papers_df.references.isnull(), 'references'].index:
    papers_df.at[row, 'references'] = []
    
papers_df_filtered = papers_df[papers_df['n_citation'] != 50]

def search_papers(user_input):
    matching_papers = {}

    for index, row in papers_df_filtered.iterrows():
        paper_id=row["id"]
        title = row["title"]
        abstract = row["abstract"]
        paper_references=row["references"]
        
        if isinstance(title, str) and user_input.lower() in title.lower():
            matching_papers[paper_id] = paper_references
        elif isinstance(abstract, str) and user_input.lower() in abstract.lower():
            matching_papers[paper_id] = paper_references
    return matching_papers


def create_graph(matching_papers):
    G = nx.Graph()
    
    # Nodes
    for _id in matching_papers.keys():
        G.add_node(_id)
    
    # Edges
    for _id, references in matching_papers.items():
        for reference in references:
            G.add_edge(_id, reference)
    return G
    
def girvan_newman(G):
    communities = list(nx.connected_components(G))
    while len(communities) == 1:
        betweenness = nx.edge_betweenness_centrality(G)
        max_betweenness = max(betweenness.values())
        edges_to_remove = [edge for edge, betweenness_value in 
                           betweenness.items() if betweenness_value == max_betweenness]
        G.remove_edges_from(edges_to_remove)
        communities = list(nx.connected_components(G))
    return communities

def compute_modularity(G, communities):
    m = len(G.edges())
    modularity = 0

    for community in communities:
        for i in community:
            for j in community:
                A_ij = 1 if G.has_edge(i, j) else 0
                k_i = len(list(G.neighbors(i)))
                k_j = len(list(G.neighbors(j)))
                modularity += (A_ij - (k_i * k_j) / (2 * m))

    modularity /= (2 * m)

    return modularity

def degree_centrality_of_community(G, community):
    subgraph = G.subgraph(community)
    degree_centrality = nx.degree_centrality(subgraph)
    return degree_centrality

def influential_papers_in_community(degree_centrality, threshold):
    influential_papers = {node for node, centrality in degree_centrality.items() if 
                          centrality >= threshold}
    return influential_papers
    
def main():
    user_input=input("prompt: ")
    matching_papers=search_papers(user_input)
    G = create_graph(matching_papers)
    communities = girvan_newman(G)
    modularity = compute_modularity(G, communities)
    print(modularity)
    threshold=0.5
    for community in communities:
        degree_centrality = degree_centrality_of_community(G, community)
        highly_influential_papers = influential_papers_in_community(degree_centrality, threshold)
        print(highly_influential_papers)
        
        
if __name__=='__main__':
    main()