import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colors
from util import graph
from util import tailorshop

# Draws a given graph into a given axis
def _draw_graph(G, ax, cmap=None, scale=False, 
                alpha=0.9, vmin=-1, vmax=1, 
                font_size=12, label_distance_x=1.06, label_distance_y=1.06):
    if cmap is None:
        cmap = plt.colormaps.get_cmap("RdBu")
    else:
        cmap = plt.colormaps.get_cmap(cmap)

    # Obtain node positions
    pos = nx.circular_layout(G)
    
    # Check which edges are multidirectional to create arc arrows
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

    # Determine colors for each node
    node_color_list = []
    for node in G.nodes:
        if node in tailorshop.controllable_variables:
            node_color_list.append("#62CA9B")
        else:
            node_color_list.append("#62A0CA")

    # Determine label positions
    lpos = {k: [v[0] * label_distance_x, v[1] * label_distance_y] for k, v in pos.items() }

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, node_size=300)
    labels = nx.draw_networkx_labels(G, lpos, ax=ax, font_size=font_size, font_color="#000")
    
    # Style label text
    for key,t in labels.items():
        t.set_va('center')
        t.set_ha('center')
        t.set_text(t.get_text().replace("_", "\n"))

    # Obtain color map based on shared min and max
    cnorm = None
    if scale:
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cnorm = lambda x: colors.Normalize(vmin=-1, vmax=1)(0.5) if x >= 0 \
                    else colors.Normalize(vmin=-1, vmax=1)(-0.5)

    # Draw edges sorted by weight (to put important edges on top)
    for idx, (u, v, d) in enumerate(
        sorted(G.edges(data=True), key=lambda x: abs(x[2]["weight"]))):

        # Determine color and edge style (dotted vs continuous) of edge
        color = cmap(cnorm(d["weight"]))
        arrowstyle = "-|>"
        edge_style = ':' if v in tailorshop.controllable_variables else "-"
        
        # Draw edge as an arc (multidirectional) or straight (unidirectional)
        edge = {(u, v) : d["weight"]}
        if (u, v) not in multidirectional:
            nx.draw_networkx_edges(G, pos, ax=ax, style=edge_style, edgelist=edge.keys(), edge_color=color, alpha=alpha, width=2.0, arrows=True, arrowstyle=arrowstyle)
        else:
            x1, y1 = pos[u][0], pos[u][1]
            x2, y2 = pos[v][0], pos[v][1]
            distance = multidirectional[(u, v)]
            distance -= min_multi_dist
            if max_multi_dist != min_multi_dist:
                distance /= (max_multi_dist - min_multi_dist)
    
            # Determine radius based on the length of the arrow
            rad = (1 - distance) * 0.2 + distance * 0.000001
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=edge.keys(), 
                alpha=alpha, 
                style=edge_style, 
                connectionstyle=f'arc3, rad={rad}', 
                edge_color=color, 
                width=2.0, 
                arrows=True, 
                arrowstyle=arrowstyle)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Main method for drawing a given graph
def draw_graph(G, title, scale=False, vmin=-1, vmax=1):
    plt.rcParams["font.family"] = "Roboto"
    plt.figure(figsize=(5,5))
    _draw_graph(G, plt.gca(), scale=scale, vmin=vmin, vmax=vmax, font_size=10, 
        label_distance_x=1.08, label_distance_y=1.08)
    plt.tight_layout()
    plt.box(False)
    plt.savefig("graph_{}.pdf".format(title), bbox_inches='tight', pad_inches = 0, format="pdf")
    plt.show()

# Obtain min and max of the weights in a graph
def get_vmin_vmax(G, symmetric=True):
    vmin = 0
    vmax = 0
    for (u, v, d) in G.edges(data=True):
        vmin = min(vmin, d["weight"])
        vmax = max(vmax, d["weight"])
    if symmetric:
        abs_max = max(vmax, abs(vmin))
        vmax = abs_max
        vmin = - abs_max
    return vmin, vmax

# Load Tailorshop truth graph
ts_graph = graph.get_networkx_graph(graph.get_graph_from_nodeslist(tailorshop.edges))

# Draw the true graph
draw_graph(ts_graph, "TS_Truth", scale=False)

# Load participants' graphs
df = pd.read_csv("../data/knowledge_data.csv")

# Filter missing information
df = df[(df["knowledge before"] != "[]") & (df["knowledge after"] != "[]")]

# Build aggregated graphs
before, avg_before = graph.get_aggregate_graph(df, "knowledge before")
after, avg_after = graph.get_aggregate_graph(df, "knowledge after")

# Filter graphs for visual clarity with a threshold of at least 5% of participants
filtered_before = graph.get_networkx_graph(graph.filter_graph(before, weight_threshold=0.05))
filtered_after = graph.get_networkx_graph(graph.filter_graph(after, weight_threshold=0.05))

# Determine shared min and max values for edges (between before and after)
vmin, vmax = get_vmin_vmax(filtered_before)
_vmin, _vmax = get_vmin_vmax(filtered_after)
abs_max = np.max(np.abs([vmin, vmax, _vmin, _vmax]))
vmax = abs_max
vmin = -abs_max

# Draw graphs
draw_graph(filtered_before, "All_paths_Before", scale=True, vmin=vmin, vmax=vmax)
draw_graph(filtered_after, "All_paths_After", scale=True, vmin=vmin, vmax=vmax)
