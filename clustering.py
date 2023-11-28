import itertools
import pickle 
import networkx as nx
from dataloader import *
from collections import deque, defaultdict
from tqdm import tqdm
import community as community_louvain

def bfs_shortest_paths(G, root):
    shortest_paths_dict = {root: [[root]]}
    queue = deque([(root, [root])])

    while queue:
        s, path = queue.popleft()

        for neighbor in G.neighbors(s):
            new_path = path + [neighbor]
            old_path = shortest_paths_dict.get(neighbor, [[None] * (len(new_path) + 1)])

            if len(new_path) == len(old_path[0]):
                shortest_paths_dict[neighbor].append(new_path)
            elif len(new_path) < len(old_path[0]):
                shortest_paths_dict[neighbor] = [new_path]
                queue.append((neighbor, new_path))

    return shortest_paths_dict

def edge_betweenness_centrality(G):
    edge_betweenness = defaultdict(float)

    for node in G.nodes():
        shortest_paths_dict = bfs_shortest_paths(G, node)

        for paths in shortest_paths_dict.values():
            for path in paths:
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    edge_betweenness[edge] += 1.0

    return edge_betweenness

def girvan_newman_directed(G):
    G_copy = G.copy()
    communities = list(nx.weakly_connected_components(G_copy))
    results = {0: communities}
    
    step = 1
    
    while G_copy.number_of_edges() > 0:
        edge_betweenness = edge_betweenness_centrality(G_copy)
        max_betweenness = max(edge_betweenness.values())
        highest_betweenness_edges = [edge for edge, value in edge_betweenness.items() if value == max_betweenness]
        G_copy.remove_edges_from(highest_betweenness_edges)
        components = list(nx.weakly_connected_components(G_copy))
        results[step] = components
        step += 1
    
    return results

def modularity(G, clusters_list):
    Q = 0
    m = len(list(G.edges()))
    for aCommunity in clusters_list:
        print("aCommunity", aCommunity)
        for v in list(aCommunity):
            for w in list(aCommunity):
                if v != w:
                    avw = 1 if (v,w) in list(G.edges()) or (w,v) in list(G.edges()) else 0               
                    new_term = avw - (G.degree(v)*G.degree(w))/(2*m)
                    Q += new_term
    return Q/(2*m)

def compute_modularity_for_all_communities(G, all_communities):
    result = []
    t = tqdm(total=len(list(all_communities.values())))
    for aCommunityRepartition in list(all_communities.values()):
        t.update()
        aModularity = modularity(G, aCommunityRepartition)
        result.append(
            [aCommunityRepartition, aModularity]
        )
    t.close    
    return result


def main():

    '''
    G= nx.DiGraph()
    all_nodes = ['A','B','C','D','E','F','G','H']
    G.add_nodes_from(all_nodes)
    all_edges = [
        ('E','D'),('E','F'),
        ('F','G'),('D','G'),('D','B'),('B','A'),('B','C'),('A','H'),
        ('D','F'),('A','C')
    ]
    G.add_edges_from(all_edges)
    '''
    
    with open("sub_graph.pkl", 'rb') as f:
        G = pickle.load(f)
    f.close()
    G = G.to_undirected()
    
    all_communities = []
    print('Finding communities...')
    com = nx.community.girvan_newman(G)
    
    print('Computing modularity...')
    
    # compute the best partition
    partition = community_louvain.best_partition(G)

    # compute modularity
    mod = community_louvain.modularity(partition, G)

    number_of_communities = len(set(partition.values()))
    print('Using the Louvain algortihm we identified', number_of_communities, 'communities')
    
    '''
    with tqdm() as t:
        for c in com:
            mod = nx.community.modularity(G, c)
            all_communities.append([list(map(set, c)), mod])
            t.update()
    '''
    #all_communities.sort(key = lambda x : x[1], reverse=True)
    #print(all_communities[0])
    
    #with open("all_clusters_with_modularity.pkl", 'wb') as f:
    #    pickle.dump(all_communities, f)    
    
        
if __name__=='__main__':
    main()
    


    #all_communities = {i: list(map(set, community)) for i, community in enumerate(com)}
    #nx.community.modularity(G, all_communities)
    #all_clusters_with_modularity = compute_modularity_for_all_communities(G, all_communities)
    #for communities in itertools.islice(all_communities, 2):
    #    print(tuple(sorted(c) for c in communities))
    #girvan_newman_directed(G)