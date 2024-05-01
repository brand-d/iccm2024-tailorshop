import pandas as pd
import json
import numpy as np
import networkx as nx
from scipy.stats import kendalltau
from util import graph
from util import tailorshop

def get_importance_dict(G):
    occurances = []
    for node in tailorshop.all_variables:
        all_paths = nx.all_simple_paths(G, node, "Comp_Val", cutoff=7)
        for path in all_paths:
            occurances.extend(path[1:])
    importances = dict(zip(*np.unique(occurances, return_counts=True)))
    return importances

def agg_importance_dicts(dicts):
    result = {}
    for node in tailorshop.all_variables:
        result[node] = 0
        for d in dicts:
            if node in d:
                result[node] += d[node] / len(dicts)
    return result

def get_cv_distance(G, node):
    dist_cv = 0
    try:
        dist_cv = nx.shortest_path_length(G, source=node, target="Comp_Val")
    except:
        return np.nan
    return dist_cv

def get_indiv_graphs(df):
    befores = []
    afters = []
    for _, row in df.iterrows():
        after_graph = graph.get_graph_from_nodeslist( \
            json.loads(row["knowledge after"]), replacement=True)
        before_graph = graph.get_graph_from_nodeslist( \
            json.loads(row["knowledge before"]), replacement=True)
        
        befores.append(before_graph)
        afters.append(after_graph)
    return befores, afters

# Load data and remove entries without graph information
df = pd.read_csv("../data/knowledge_data.csv")
df = df[(df["knowledge before"] != "[]") & (df["knowledge after"] != "[]")]

# Obtain the Tailorshop ground truth graph
ts_graph = graph.get_graph_from_nodeslist(tailorshop.edges)
ts_graph_nx = graph.get_networkx_graph(ts_graph)

# Obtain graphs for before and after engaging with the TS simulation
before, avg_before = graph.get_aggregate_graph(df, "knowledge before")
after, avg_after = graph.get_aggregate_graph(df, "knowledge after")

# Print information about edge count
print("----------------------------------------Edges----------------------------------------")
print("Edges in combined graph:\n\tBefore: {}\n\tAfter: {}\n\tTS Truth: {}".format(
    len(before), len(after), len(tailorshop.edges)))
print("Avg. edges per participant:\n\tBefore: {}\n\tAfter: {}".format(
    np.round(avg_before, 2), np.round(avg_after, 2)))

# Obtain individual graphs 
before_indiv, after_indiv = get_indiv_graphs(df)

# Generate nx graph for individual graphs
before_indiv_nx = [graph.get_networkx_graph(x) for x in before_indiv]
after_indiv_nx = [graph.get_networkx_graph(x) for x in after_indiv]

print("--------------------------------------Diameter---------------------------------------")
print("Diameter (longest of the shortest paths between all nodes)")
print("\tBefore (avg): {}\n\tAfter (avg): {}\n\tTS Truth: {}".format(
    np.round(np.mean([graph.diameter(x) for x in before_indiv_nx]), 3),
    np.round(np.mean([graph.diameter(x) for x in after_indiv_nx]), 3),
    graph.diameter(ts_graph_nx)))
print()

print("-----------------------------------Connections---------------------------------------")
# Collect the number of ingoing and outgoing edges for each node
outmeans_before = []
outmeans_after = []
outmeans_ts = []
inmeans_before = []
inmeans_after = []
inmeans_ts = []
for node in tailorshop.all_variables:
    inmeans_before.append(np.mean(
        [len(list(x.predecessors(node))) for x in before_indiv_nx]))
    inmeans_after.append(np.mean(
        [len(list(x.predecessors(node))) for x in after_indiv_nx]))
    outmeans_before.append(np.mean(
        [len(list(x.successors(node))) for x in before_indiv_nx]))
    outmeans_after.append(np.mean(
        [len(list(x.successors(node)))for x in after_indiv_nx]))
    outmeans_ts.append(len(list(ts_graph_nx.successors(node))))
    inmeans_ts.append(len(list(ts_graph_nx.predecessors(node))))

# Print results
print("Ingoing edges:")
print(f"\tBefore:\
    \tmean={np.round(np.mean(inmeans_before), 2)},\
    \tSD={np.round(np.std(inmeans_before), 2)},\
    \tmin={np.round(np.min(inmeans_before), 2)}, \
    \tmax={np.round(np.max(inmeans_before), 2)}")
print(f"\tAfter:\
    \tmean={np.round(np.mean(inmeans_after), 2)},\
    \tSD={np.round(np.std(inmeans_after), 2)},\
    \tmin={np.round(np.min(inmeans_after), 2)},\
    \tmax={np.round(np.max(inmeans_after), 2)}")
print(f"\tTS Truth:\
    \tmean={np.round(np.mean(inmeans_ts), 2)},\
    \tSD={np.round(np.std(inmeans_ts), 2)},\
    \tmin={np.round(np.min(inmeans_ts), 2)},\
    \tmax={np.round(np.max(inmeans_ts), 2)}")
print()
print("Outgoing edges:")
print(f"\tBefore:\
    \tmean={np.round(np.mean(outmeans_before), 2)},\
    \tSD={np.round(np.std(outmeans_before), 2)},\
    \tmin={np.round(np.min(outmeans_before), 2)},\
    \tmax={np.round(np.max(outmeans_before), 2)}")
print(f"\tAfter:\
    \tmean={np.round(np.mean(outmeans_after), 2)},\
    \tSD={np.round(np.std(outmeans_after), 2)},\
    \tmin={np.round(np.min(outmeans_after), 2)},\
    \tmax={np.round(np.max(outmeans_after), 2)}")
print(f"\tTS Truth:\
    \tmean={np.round(np.mean(outmeans_ts), 2)},\
    \tSD={np.round(np.std(outmeans_ts), 2)},\
    \tmin={np.round(np.min(outmeans_ts), 2)},\
    \tmax={np.round(np.max(outmeans_ts), 2)}")
print()


print("-----------------------------------Node Importances-----------------------------------")
importances_before = agg_importance_dicts([get_importance_dict(x) for x in before_indiv_nx])
importances_after = agg_importance_dicts([get_importance_dict(x) for x in after_indiv_nx])
importances_ts = agg_importance_dicts([get_importance_dict(ts_graph_nx)])

print("{}\t{}\t{}\t{}".format(
    "Node".ljust(12, " "),
    "Before".ljust(12, " "),
    "After".ljust(12, " "), 
    "TS Truth".ljust(12, " ")))
for node in tailorshop.derived_variables:
    imp_before = np.round(importances_before[node], 2)
    imp_after = np.round(importances_after[node], 2)
    imp_ts = int(importances_ts[node])
    print("{}\t{}\t{}\t{}".format(
        node.ljust(12, " "),
        str(imp_before).ljust(12, " "),
        str(imp_after).ljust(12, " "),
        str(imp_ts).ljust(12, " ")))
print()

# Sort derived vars by importance to obtain rank
before_list = sorted(
    [(k, v) for k, v in importances_before.items() 
        if k in tailorshop.derived_variables],
    key=lambda x: x[1], reverse=True)
before_list = [x[0] for x in before_list]

ts_list = sorted(
    [(k, v) for k, v in importances_ts.items() 
        if k in tailorshop.derived_variables],
    key=lambda x: x[1], reverse=True)
ts_list = [x[0] for x in ts_list]

# Calculate Kendalls Tau for Rank correlation
rnk_before = [before_list.index(x) for x in tailorshop.derived_variables]
rnk_ts = [ts_list.index(x) for x in tailorshop.derived_variables]

tau, p_val = kendalltau(
    rnk_before,
    rnk_ts, 
    method = 'exact')

print("Correspondance (Before vs TS Truth):")
print("Kendall's tau={}, p={}".format(
    np.round(tau, 3),
    np.round(p_val, 3)))
