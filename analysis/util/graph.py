import pandas as pd
import json
import numpy as np
import networkx as nx
from networkx.algorithms import approximation as approx
import matplotlib.pyplot as plt
from matplotlib import colors
from . import tailorshop

def draw_graph(G, ax, cmap=None, scale=False, 
               alpha=0.9, vmin=-1, vmax=1,
               font_size=12, label_distance=1.06):
    if cmap is None:
        cmap = plt.colormaps.get_cmap("RdBu")
    else:
        cmap = plt.colormaps.get_cmap(cmap)

    pos = nx.circular_layout(G)
    
    multidirectional = {}
    min_multi_dist = 2
    max_multi_dist = 0
    multidir_tmp = set()
    for (u, v, d) in G.edges(data=True):
        if (u, v) in multidir_tmp:
            x1, y1 = pos[u][0], pos[u][1]
            x2, y2 = pos[v][0], pos[v][1]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            multidirectional[(u, v)] = distance
            multidirectional[(v, u)] = distance
            max_multi_dist = max(max_multi_dist, distance)
            min_multi_dist = min(min_multi_dist, distance)
        multidir_tmp.add((u, v))
        multidir_tmp.add((v, u))
    multidir_tmp = None

    node_color_list = []
    for node in G.nodes:
        if node in tailorshop.controllable_variables:
            node_color_list.append("#1FB471")
        else:
            node_color_list.append("#1f78b4")

    theta = {k: np.arctan2(v[1], v[0]) * 180/np.pi for k, v in pos.items() }
    lpos = {k: [v[0] * label_distance, v[1] * label_distance] for k, v in pos.items() }

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list)
    labels = nx.draw_networkx_labels(G, lpos, ax=ax, font_size=font_size)
    
    for key,t in labels.items():
        angle = theta[key] + 270
        if 100 <= angle <= 270:
            angle += 180
        t.set_va('center')
        t.set_ha('center')
        t.set_rotation(angle)
        t.set_rotation_mode('anchor')

    cnorm = None
    if scale:
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cnorm = lambda x: colors.Normalize(vmin=-1, vmax=1)(0.5) if x >= 0 \
                    else colors.Normalize(vmin=-1, vmax=1)(-0.5)

    for idx, (u, v, d) in enumerate(sorted(G.edges(data=True), key=lambda x: abs(x[2]["weight"]))):
        edge = {(u, v) : d["weight"]}
        
        color = cmap(cnorm(d["weight"]))
        arrowstyle = "-|>"
        edge_style = ':' if v in tailorshop.controllable_variables else "-"
        
        if (u, v) not in multidirectional:
            nx.draw_networkx_edges(G, pos, ax=ax, style=edge_style, edgelist=edge.keys(), edge_color=color, alpha=alpha, width=2.0, arrows=True, arrowstyle=arrowstyle)
        else:
            x1, y1 = pos[u][0], pos[u][1]
            x2, y2 = pos[v][0], pos[v][1]
            distance = multidirectional[(u, v)]
            distance -= min_multi_dist
            if max_multi_dist != min_multi_dist:
                distance /= (max_multi_dist - min_multi_dist)
    
            rad = (1 - distance) * 0.2 + distance * 0.000001
            nx.draw_networkx_edges(G, pos, ax=ax, style=edge_style, edgelist=edge.keys(), alpha=alpha, connectionstyle=f'arc3, rad={rad}', edge_color=color, width=2.0, arrows=True, arrowstyle=arrowstyle)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def avg_cosine_sim(A, B):
    cosine_sims = []
    for i in range(A.shape[0]):
        norm = (np.linalg.norm(A[i]) * np.linalg.norm(B[i]))
        if norm == 0:
            cosine_sims.append(0)
        else:
            cos_sim = np.dot(A[i], B[i]) / norm
            cosine_sims.append(cos_sim)
    return np.mean(cosine_sims)

def euclid_sim(A, B):
    diff = A - B
    frob_norm = np.linalg.norm(diff, ord="fro")
    
    max_matrix = np.full_like(diff, 2)
    max_matrix_frob = np.linalg.norm(max_matrix, ord="fro")
    return 1 - (frob_norm / max_matrix_frob)

def get_graph_from_nodeslist(nodes, replacement=False):
    agg_graph = {}
    keyset = set()
    for edge in nodes:
        start, label, end = edge
        if replacement:
            start = tailorshop.node_replacement_map[start]
            end = tailorshop.node_replacement_map[end]
        key = (start, end)
        if key in keyset:
            print("Warning: Duplicate key:", key)
        keyset.add(key)
        weight = 1.0 if label == "+" else -1.0
        if key not in agg_graph:
            agg_graph[key] = 0
        agg_graph[key] += weight
    return agg_graph

def get_aggregate_graph(df, column):
    agg_graph = {}
    num_part = len(df)
    lengths = []
    for _, participant in df.iterrows():
        keyset = set()
        graph = json.loads(participant[column])
        lengths.append(len(graph))

        for edge in graph:
            start, label, end = edge
            start = tailorshop.node_replacement_map[start]
            end = tailorshop.node_replacement_map[end]
            key = (start, end)
            
            if key in keyset:
                print("Warning: Duplicate key:", key)
            keyset.add(key)
            
            weight = 1.0 if label == "+" else -1.0
            
            if key not in agg_graph:
                agg_graph[key] = 0
            agg_graph[key] += weight / num_part
    return agg_graph, np.mean(lengths)

def get_adjacency_matrix(graph):
    nodes = list(tailorshop.all_variables)
    mat = np.zeros((len(nodes), len(nodes)))
    for edge, weight in graph.items():
        start, target = edge
        start_idx = nodes.index(start)
        end_idx = nodes.index(target)
        mat[start_idx, end_idx] = weight
    return mat

def from_adjacency_matrix(mat, threshold=0):
    nodes = list(tailorshop.all_variables)
    graph = {}
    for start_idx, start_node in enumerate(nodes):
        for end_idx, end_node in enumerate(nodes):
            weight = mat[start_idx, end_idx]
            if abs(weight) > threshold:
                graph[(start_node, end_node)] = weight
    return graph

def filter_graph(graph, weight_threshold=0, forbidden_ends=None):
    filtered = {k: v for k, v in graph.items() if abs(v) >= weight_threshold}
    if forbidden_ends is not None:
        filtered = {k: v for k, v in graph.items() if k[1] not in forbidden_ends}
    return filtered

def get_total_connection_weights(G, node, target="Comp_Val", cutoff=7):
    if cutoff is None:
        cutoff = get_longest_shortest_path_length(G, target)
    all_paths = nx.all_simple_edge_paths(G, node, target, cutoff=cutoff)
    result = []
    for path in all_paths:
        total_weight = 1
        for u, v in path:
            weight = G.get_edge_data(u, v, default=0)["weight"]
            total_weight *= weight 
        result.append((path, total_weight))
    return result

def get_networkx_graph(graph):
    G = nx.DiGraph()
    G.add_nodes_from(tailorshop.all_variables)
    
    weights = []
    for edge, weight in graph.items():
        G.add_edge(edge[0], edge[1], weight=weight)
        weights.append(weight)
    
    return G

def get_longest_shortest_path_length(G, target):
    longest = 0
    for node in tailorshop.all_variables:
        if node == target:
            continue
        try:
            path_length = nx.shortest_path_length(G, source=node, target=target)
            longest = max(longest, path_length)
        except (nx.exception.NetworkXNoPath):
            print("No path to {} found for {}".format(target, node))
            pass
    return longest

def longest_simple_paths(graph, source, target):
    longest_paths = []
    longest_path_length = 0
    for path in nx.all_simple_paths(graph, source=source, target=target):
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_paths.clear()
            longest_paths.append(path)
        elif len(path) == longest_path_length:
            longest_paths.append(path)
    return longest_paths

def get_reduced_graph(graph, target="Company_val", method="longest_shortest", verbose=False, get_nx=True):
    reduced_graph = {}
    all_edges = graph.edges(data=True)
    
    connection_depth = 0
    if method == "longest_shortest" or method == "all_longest_shortest":
        connection_depth = get_longest_shortest_path_length(graph, target)
        if verbose:
            print("Longest shortest path to {}: {}".format(target, connection_depth))
    
    for node in tailorshop.all_variables:
        all_paths = None
        if method == "shortest":
            all_paths = nx.algorithms.all_shortest_paths(graph, source=node, target=target)
            if verbose:
                print("For {}:".format(node))
        elif method == "longest_shortest":
            all_paths = nx.algorithms.all_simple_paths(graph, source=node, target=target, cutoff=connection_depth)
            if verbose:
                print("For {}:".format(node))
        elif method == "longest":
            all_paths = longest_simple_paths(graph, node, target)
        elif method == "all_longest_shortest":
            all_paths = []
            for cur_target in tailorshop.all_variables:
                all_paths.extend(list(nx.algorithms.all_simple_paths(graph, source=node, target=cur_target, cutoff=connection_depth)))
        else:
            raise ValueError("Method has to be one of ['shortest', 'longest_shortest', 'longest', 'all_longest_shortest'].")
        for path in all_paths:
            if verbose:
                print("    ", path)
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                relevant_edge = next(filter(lambda x: (x[0] == start) and (x[1] == end), all_edges))
                new_edge = (start, end)
                if new_edge not in reduced_graph:
                    reduced_graph[new_edge] = relevant_edge[2]["weight"]
        if verbose:
            print()
    if get_nx:
        return get_networkx_graph(reduced_graph)
    else:
        return reduced_graph

def diameter(G):
    result = 0
    for start in tailorshop.all_variables:
        for target in tailorshop.all_variables:
            if start == target:
                continue
            pl = 0
            try:
                pl = nx.shortest_path_length(G, source=start, target=target)
            except (nx.exception.NetworkXNoPath):
                pass
            result = max(result, pl)
    return result

def get_paths_between(G, start_node, target_node, length=3):
    paths = None
    paths = nx.all_simple_paths(G, start_node, target_node, cutoff=length)
    paths = sorted(paths, key=lambda x: len(x))

    if len(paths) > 0:
        print("    {}:".format(start_node))

        for path in paths:
            connection = ""
            all_edges = G.edges(data=True)
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                relevant_edge = next(filter(lambda x: (x[0] == start) and (x[1] == end), all_edges))
                symbol = " +> " if relevant_edge[2]["weight"] > 0 else " -> "
                connection += "{}{}".format(symbol, end)

            print("        {}{}".format(path[0], connection))
            lengths.append(len(path) - 1)
    else:
        print("    {}: -".format(start_node))
