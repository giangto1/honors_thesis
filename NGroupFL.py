'''Need package networkx'''


class Node:
    '''
    define the properties of node in G, its neighbors and indices of edges
    '''
    neighbor_left = []
    neighbor_right = []
    neighbor_right_idx = []
    neighbor_left_idx = []

    def __init__(self, node, G):
        edgelist = list(G.edges())
        self.node = node
        if node in G.nodes():
            self.neighbor_right = [j for j in list(
                G.neighbors(node)) if j > node]
            self.neighbor_left = [j for j in list(
                G.neighbors(node)) if j < node]

        self.neighbor_right_idx = [edgelist.index(
            (node, k)) for k in self.neighbor_right]
        self.neighbor_left_idx = [edgelist.index(
            (k, node)) for k in self.neighbor_left]
