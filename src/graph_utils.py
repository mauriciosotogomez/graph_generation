# Packages

import pandas as pd
import networkx as nx
import numpy as np
from math import ceil
from math import log
from itertools import combinations


import matplotlib.pyplot as plt

# Erdos-Renyi

def generate_erdos_renyi_graph (n=10, p = 0.3, num_node_types=3) :
    """
    generate a erdos renyi graph
    n : number of nodes
    p : probability of the existence of an edge
    num_classes : number of the node classes (node types are assigned randomly)
    feature_dim : dimension of the (0/1) tensor used as input feature. If 0 (def) a one-hot encoding is generated.

    Example:
    er_edges, er_nodes = generate_erdos_renyi_graph()        
    """
    er = nx.erdos_renyi_graph(n, p)
    edges_df = nx.to_pandas_edgelist(er)
    nodes_df = pd.DataFrame(list(er.nodes()), columns=['name'])
    nodes_df['type'] = np.random.randint(0, num_node_types, nodes_df.shape[0])

    return edges_df, nodes_df

# Babarasi-Albert (Preferential Attachment)

def generate_babarasi_albert (n=10, m=5, num_node_types=3) :
    """
    generate a Babarasi Albert graph 
    
    Parameters
    ----------
    n   int
        Number of nodes
    m   int
        Number of edges to attach from a new node to existing nodes
    
    """
    ba = nx.barabasi_albert_graph(n, m)
    edges_df = nx.to_pandas_edgelist(ba)
    nodes_df = pd.DataFrame(list(ba.nodes()), columns=['name'])
    nodes_df['type'] = np.random.randint(0, num_node_types, nodes_df.shape[0])

    return edges_df, nodes_df


# Power-Law

def generate_power_law_from_file (adjacency_list_file="degs", num_node_types=10) :
    """
    This method generates a power law graph using the generator described in 
    https://www-complexnetworks.lip6.fr/~latapy/FV/generation.html
    The method assume the existence of a file containing the adjacency list  
    """

    # Read the adjacency list from a CSV file
    df_adjacency = pd.read_csv(adjacency_list_file, header=None)

    # Create an empty DataFrame for the edge and node list
    edges_df = pd.DataFrame(columns=['source', 'target'])
    # Create a set to store nodes
    nodes_set = set()

    # Iterate over each row in the adjacency list
    for index, row in df_adjacency.iterrows():
        row_split = str(row[0]).split(' ')[:-1]
        nodes_set.update(row_split)
        source = row_split[0]
        neighbors = row_split[1:]
        for target in neighbors:
            new_edge = {'source': source, 'target': target}
            edges_df = pd.concat([edges_df, pd.DataFrame([new_edge])], ignore_index=True)

    # Create node dataframe
    num_node_types=10
    nodes_df = pd.DataFrame(nodes_set,columns=['name'])
    nodes_df['type'] = np.random.randint(0, num_node_types, nodes_df.shape[0])

    return edges_df, nodes_df


# Regular Tree

# Cycle

def generate_cycle (n=10,n_types=5):
    """
    generate a cyce with n nodes 
    Types are assigned as  
    """
    n=n+1
    typed_edges = [[i-1,i] for i in range(2,n)]
    typed_edges.append([n-1,1])
    typed_nodes = [[i, int((n_types*i)/n)] for i in range(1,n)]

    nodes_df = pd.DataFrame(typed_nodes, columns=['name','type'])
    edges_df =  pd.DataFrame(typed_edges, columns=['source','target'])

    return edges_df, nodes_df

# Grid

def generate_grid (rows=5, columns=10):
    """
    generate a nxm grid
    rows : number of rows
    columns : number of columns
    """
    n=rows
    m=columns
    
    n_nodes_grid = n*m

    typed_nodes=[[i, int(i/m)] for i in range(n_nodes_grid)] # node type equals the row
    typed_edges=[]
    for i in range(n_nodes_grid):
        if (i+1)%m != 0 : # not the last column
            typed_edges.append([i, i+1])
        if i < (n-1)*m : # not the last a row
            typed_edges.append([i, i+m])

    nodes_df = pd.DataFrame(typed_nodes, columns=['name','type'])
    edges_df =  pd.DataFrame(typed_edges, columns=['source','target'])

    return edges_df, nodes_df


# Complete bipartite

def generate_complete_bipartite (n1=4, n2=4) :
    """
    generate a complete bipartite graph Kkn
    n1 : number of nodes in the first partition
    n2 : number of nodes in the second partition
    """
    k=n1
    n=n2
    
    typed_nodes=[[i, 0] for i in range(k)] + [[k+j, 0] for j in range(n)]
    typed_edges=[[i, j] for i in range(k) for j in range(k,k+n)]

    nodes_df = pd.DataFrame(typed_nodes, columns=['name','type'])
    edges_df =  pd.DataFrame(typed_edges, columns=['source','target'])

    return edges_df, nodes_df

def generate_complete_graph (n=20, k=4) :
    """
    generate a complete K-partite graph Knn
    k : number of groups
    n : number of nodes 
    """

    typed_nodes=[[i, int(k*i/n)] for i in range(n)]
    typed_edges=[[i, j] for i in range(n) for j in range(i)]
    
    nodes_df = pd.DataFrame(typed_nodes, columns=['name','type'])
    edges_df =  pd.DataFrame(typed_edges, columns=['source','target'])

    return edges_df, nodes_df

    
def generate_clustered_grid (rows=20,columns=10,groups=6) :
    n = rows # number of rows
    m = columns # number of columns
    k = groups  # groups
    
    n_nodes_grid = n*m

    def node_type(i) :
        return k*(i//(m*k)) + (i%m//k) 

    typed_nodes=[[i, node_type(i) ] for i in range(n_nodes_grid)] # node type 
    typed_edges=[]

    #grid edges
    for i in range(n_nodes_grid):
        if (i+1)%m != 0 : # not the last column
            typed_edges.append([i, i+1,1])  # no edge type 
        if i < (n-1)*m : # not the last a row
            typed_edges.append([i, i+m,1])  # no edge type    

    nodes_df = pd.DataFrame(typed_nodes, columns=['name','type'])
    edges_df =  pd.DataFrame(typed_edges, columns=['source','target','predicate'])
    edges_df['weight'] = np.ones(len(typed_edges))

    return edges_df, nodes_df


# Community graph

def generate_community_graph(k=4, 
                            min_size=2, 
                            max_size=100,
                            intra_prob=0.7,
                            inter_prob=0.001,
                            seed=42):
    """
    Generate a graph with k densely connected subgraphs (communities).
    
    Parameters:
    -----------
    k : int
        Number of subgraphs/communities
    min_size : int
        Minimum size of each community
    max_size : int
        Maximum size of each community
    intra_prob : float
        Probability of edge within a community (dense connections)
    inter_prob : float
        Probability of edge between communities (sparse connections)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    nodes_df : DataFrame
        DataFrame with columns ['node', 'type']
    edges_df : DataFrame
        DataFrame with columns ['source', 'target']
    """
    np.random.seed(seed)
    
    # Generate community sizes
    community_sizes = np.random.randint(min_size, max_size + 1, size=k)
    
    nodes = []
    edges = []
    node_id = 0
    
    # Store node ranges for each community
    community_ranges = []
    
    # Generate nodes and intra-community edges
    for comm_idx in range(k):
        comm_size = community_sizes[comm_idx]
        comm_start = node_id
        comm_type = f'community_{comm_idx + 1}'
        
        # Create nodes for this community
        comm_nodes = []
        for i in range(comm_size):
            node_name = f'node_{node_id}'
            nodes.append({'name': node_name, 'type': comm_type})
            comm_nodes.append(node_name)
            node_id += 1
        
        community_ranges.append(comm_nodes)
        
        # Create dense intra-community edges
        for node1, node2 in combinations(comm_nodes, 2):
            if np.random.random() < intra_prob:
                edges.append({'source': node1, 'target': node2})
    
    # Create sparse inter-community edges
    for i in range(k):
        for j in range(i + 1, k):
            # Connect nodes between community i and j
            for node1 in community_ranges[i]:
                for node2 in community_ranges[j]:
                    if np.random.random() < inter_prob:
                        edges.append({'source': node1, 'target': node2})
    
    # Create dataframes
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    return edges_df, nodes_df

# Generate Tree

def generate_regular_tree(tree_depth=4, degree=3):
    """
    Generate a graph with k densely connected subgraphs (communities).
    
    Parameters:
    -----------
    tree_depth : int
        Tree depth
    degree : int
        Number of descendants 
    
    Returns:
    --------
    nodes_df : DataFrame
        DataFrame with columns ['node', 'type']
    edges_df : DataFrame
        DataFrame with columns ['source', 'target']
    """

    n_nodes_tree = int((degree**(tree_depth+1)-1)/(degree-1))
    #log(30*4+degree,degree)=3.0000000000000004 !!!
    tol        = 0.0000000000000004 

    typed_nodes=[[i, ceil(log((degree-1)*i+degree,degree)-1-tol)] for i in range(n_nodes_tree)]
    typed_edges=[[int((i-1)/degree), i, ceil(log((degree-1)*i+degree,degree)-1-tol)] for i in range(1,n_nodes_tree)]
    
    nodes_df = pd.DataFrame(typed_nodes, columns=['name','type'])
    edges_df =  pd.DataFrame(typed_edges, columns=['source','target','predicate'])
    edges_df['weight'] = np.ones(len(typed_edges))

    return edges_df, nodes_df


def plot_graph(nodes_df,edges_df) :
    # Create the graph
    G = nx.Graph()
    
    # Add nodes with their type attribute
    for _, row in nodes_df.iterrows():
        G.add_node(row['name'], node_type=row['type'])
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # Get unique node types and assign colors
    unique_types = nodes_df['type'].unique()
    color_palette = plt.cm.tab20(range(len(unique_types)))
    type_to_color = dict(zip(unique_types, color_palette))
    node_colors = [type_to_color[G.nodes[node]['node_type']] for node in G.nodes()]
    
    # Plot the graph
    nx.draw_networkx(G,node_color=node_colors,alpha=.7,pos=nx.kamada_kawai_layout(G),node_size=100,with_labels=False)

    return G

def plot_loghist(x, bins):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    plt.yscale('log')

