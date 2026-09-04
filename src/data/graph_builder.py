import numpy as np

# Sensor node indices
NODE_NAMES = ['P1', 'A1', 'H1', 'H2', 'A2', 'P2']
NODE_MAP = {name: i for i, name in enumerate(NODE_NAMES)}

def get_network_graph(topology='Branched'):
    """
    Constructs the physical pipe network graph G = (V, E)
    Nodes (6):
      0: P1 (Supply Line entrance)
      1: A1 (Tee Connection 1 valve branch)
      2: H1 (Prototype Hydrant 1 / residential mid-pipe)
      3: H2 (Prototype Hydrant 2 / intersection hydrant)
      4: A2 (Tee Connection 2 valve branch)
      5: P2 (Farthest corner downstream pipe)
    """
    num_nodes = 6
    # Edge list (bidirectional pipe connections)
    edges = [
        (0, 1), (1, 0), # P1 <-> A1
        (1, 2), (2, 1), # A1 <-> H1
        (2, 3), (3, 2), # H1 <-> H2 (Middle Pipe with Leak Location)
        (3, 4), (4, 3), # H2 <-> A2
        (4, 5), (5, 4), # A2 <-> P2
    ]
    
    if topology == 'Looped':
        # Add loop-closing edge
        edges.extend([(5, 0), (0, 5), (5, 1), (1, 5)])
        
    edge_index = np.array(edges, dtype=np.int64).T # Shape (2, num_edges)
    
    # Adjacency matrix with self-loops
    adj = np.eye(num_nodes, dtype=np.float32)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        
    # Symmetrically normalized adjacency: D^(-1/2) * A * D^(-1/2)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    adj_norm = d_mat @ adj @ d_mat
    
    return {
        'num_nodes': num_nodes,
        'node_names': NODE_NAMES,
        'edge_index': edge_index,
        'adj': adj,
        'adj_norm': adj_norm
    }
