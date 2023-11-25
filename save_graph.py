from matplotlib import pyplot as plt
import networkx as nx


def save_graph(graph, file_name):
    # initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis("off")
    fig = plt.figure(1)
    print("Creating random layout")
    pos = nx.random_layout(graph)
    # nx.draw_networkx_nodes(graph, pos)
    print("adding layers")
    nx.draw_networkx_edges(graph, pos)
    # nx.draw_networkx_labels(graph, pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    print("saving")
    plt.savefig(file_name, bbox_inches="tight")
