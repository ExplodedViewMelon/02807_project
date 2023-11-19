
def calculate_keywords_with_highest_centrality_and_citations(community_paper_ids, papers, graph, num_top_papers=3, consider_citations=False):
    # TF-IDF scores for keywords in papers within the community
    paper_texts = [papers[paper_id] for paper_id in community_paper_ids]
    tfidf = TfidfVectorizer().fit_transform(paper_texts)

    # TF-IDF vectors for community papers
    tfidf_dict = {paper_id: tfidf[i] for i, paper_id in enumerate(community_paper_ids)}

    # Calculate centrality scores
    centrality = nx.eigenvector_centrality_numpy(graph, weight='tfidf')

    # Get keywords from papers with the highest centrality
    top_papers_centrality = sorted(centrality, key=centrality.get, reverse=True)[:num_top_papers]

    # Optional: Get keywords from papers with the highest number of citations
    top_papers_citations = []
    if consider_citations:
        citations = dict(graph.degree(community_paper_ids, weight='n_citations'))
        top_papers_citations = sorted(citations, key=citations.get, reverse=True)[:num_top_papers]

    # Combine papers from centrality and citations (if both options are used)
    top_papers = list(set(top_papers_centrality + top_papers_citations))

    keywords = []
    for paper_id in top_papers:
        keywords.extend(get_keywords_from_paper(papers[paper_id])) 

    return list(set(keywords))