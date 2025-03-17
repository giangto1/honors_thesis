import networkx as nx
import numpy as np

'''
necessary tools for Group Fused Lasso
'''

def coloredge(G):
    '''
    color the edges via greedy algorithm, in each group of colors
    any two edges share no vertices. coloredge(G) return the graph G containg
     edges with weights, each weight denotes the color.
    '''

    edgelist = list(G.edges())

    # print(edgelist[:3])
    for k in range(len(edgelist)):
        node1 = edgelist[k][0]
        node2 = edgelist[k][1]

        color_to_skip = []
        for i in G.neighbors(node1):
            if (i, node1) in edgelist[:k]:
                color_to_skip.append(G[i][node1]['weight'])
            if (node1, i) in edgelist[:k]:
                color_to_skip.append(G[node1][i]['weight'])

        for i in G.neighbors(node2):
            if (i, node2) in edgelist[:k]:
                color_to_skip.append(G[i][node2]['weight'])
            if (node2, i) in edgelist[:k]:
                color_to_skip.append(G[node2][i]['weight'])

        color = 0
        while color in color_to_skip:
            color += 1

        G[node1][node2]['weight'] = color

    return G


def decompose_graph(G):
    '''Decompose a graph based on colors of edges, return G0,G1. G0 is a graph
    that any two edges share no vertices, G1 is the graph containg all edges of
    G except the ones of G0'''

    # colorset return all colors
    colorset = []

    for i in [data['weight'] for node1, node2, data in G.edges(data=True)]:
        if i not in colorset:
            colorset.append(i)

    if len(colorset) == 1:
        print('E1 is empty! The alogorithm needs E1 is not empty!')

    # decompose the graph G based on the color, edgeGroup[i] return the list of
    # all edges with color i
    edgeGroup = []
    for k in colorset:
        temp = [sorted([node1, node2]) for node1, node2,
                data in G.edges(data=True) if data['weight'] == k]
        edgeGroup.append(np.array(temp))
        # print(temp)

    G1nodes = np.unique(edgeGroup[1])
    for k in range(1, len(colorset)):
        temp = np.unique(edgeGroup[k])
        G1nodes = np.unique(np.concatenate((G1nodes, temp), axis=None))

    G0 = nx.Graph()
    G1 = nx.Graph()
    G0nodes = np.unique(edgeGroup[0])

    G0.add_nodes_from(G0nodes)
    G1.add_nodes_from(G1nodes)

    G0.add_edges_from(edgeGroup[0])
    for i in range(1, len(colorset)):
        G1.add_edges_from(edgeGroup[i])

    return G0, G1


def solve_f(a, b, lam, c1, c2):
    '''argmin_x,y c1||x-a||^2+c2||y-b||^2+lam||x-y||'''
    if 2 * c1 * c2 * np.linalg.norm(a - b) <= (c1 + c2) * lam:
        return (c1 * a + c2 * b) / (c1 + c2), (c1 * a + c2 * b) / (c1 + c2)

    return a - lam * (a - b) / np.linalg.norm(a - b) / 2 / c1, b - lam * (b - a) / 2 / c2 / np.linalg.norm(b - a)


def get_neighbors_list(node, G):
    '''list of neighbors of node in E1, l1 is set of neighbors with larger vertices
    l2 is set of neighbors with smaller vertices. If node does not have a
    neighbor in G, return two empty lists.'''
    if node in G.nodes():
        l1 = [j for j in list(G.neighbors(node)) if j > node]
        l2 = [j for j in list(G.neighbors(node)) if j < node]
        return l1, l2

    return [], []


def threshold(a, b):
    '''argmin_z 1/2||z-a||^2+b||z||'''
    if np.linalg.norm(a) == 0:
        return a

    return max(np.linalg.norm(a) - b, 0) / np.linalg.norm(a) * a
