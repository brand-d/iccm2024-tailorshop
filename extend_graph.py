import pandas as pd
import json
import numpy as np
import networkx as nx
from analysis.util import graph
from analysis.util import tailorshop
from scipy import stats
import csv

def get_company_value(elem):
    final_value = json.loads(elem)["CompanyValue"]
    return final_value

def get_success_ratio(elem):
    base_value = tailorshop.default_values["Comp_Val"]
    final_value = json.loads(elem)["CompanyValue"]
    ratio = (final_value - base_value) / base_value
    return ratio

def calculate_sim_between(elem):
    participant_graph1 = graph.get_graph_from_nodeslist( \
        json.loads(elem["knowledge before"]),
        replacement=True)
    participant_graph2 = graph.get_graph_from_nodeslist( \
        json.loads(elem["knowledge after"]),
        replacement=True)
    adj_part1 = graph.get_adjacency_matrix(participant_graph1)
    adj_part2 = graph.get_adjacency_matrix(participant_graph2)
    return graph.avg_cosine_sim(adj_part1, adj_part2)

def get_crt_correctness(elem):
    correct_crt = {
        "CRT_1": "0.05",
        "CRT_2": "5",
        "CRT_3": "47",
        "CRT_4": "4",
        "CRT_5": "29",
        "CRT_6": "20",
        "CRT_7": "Geld verloren",
    }

    if elem["CRT_1"] == 5:
        elem["CRT_1"] = 0.05
    corr = 0
    for i in range(7):
        crt = f"CRT_{i+1}"
        if correct_crt[crt] == str(elem[crt]):
            corr += 1
    return corr/7

def calculate_ts_sim_for_person(elem, no_controlled_successor=False):
    participant_graph = graph.get_graph_from_nodeslist(json.loads(elem), replacement=True)
    if no_controlled_successor:
        participant_graph = graph.filter_graph(participant_graph, forbidden_ends=tailorshop.controllable_variables)
    adj_part = graph.get_adjacency_matrix(participant_graph)
    return graph.avg_cosine_sim(adj_part, adj_ts)

# Load dataset
df = pd.read_csv("data/knowledge_data.csv")

# Exclude incomplete information (missing knowledge graph)
df = df[(df["knowledge before"] != "[]") & (df["knowledge after"] != "[]")]

# Load Tailorshop graph (and adjacency matrix)
ts_graph = graph.get_graph_from_nodeslist(tailorshop.edges)
adj_ts = graph.get_adjacency_matrix(ts_graph)

# Calculate sucess (ratio of resulting company value in month 11 with starting value)
df["ts success"] = df["beforefinal ts"].apply(get_success_ratio)
df["ts success exploration"] = df["before final ts exploration"].apply(get_success_ratio)

df["company value"] = df["beforefinal ts"].apply(get_company_value)
df["debt"] = df["beforefinal ts"].apply(lambda x: json.loads(x)["BankAccount"] < 0)

# Test with NFC or CRT and self assessment
df["CRT correct"] = df[[f"CRT_{x+1}" for x in range(7)]].apply(get_crt_correctness, axis=1)
df["NFC"] = np.mean(df[[f"NFC_{x+1}" for x in range(4)]], axis=1)

# Calculate similarities with TS
df["sim before ts"] = df["knowledge before"].apply(calculate_ts_sim_for_person)
df["sim after ts"] = df["knowledge after"].apply(calculate_ts_sim_for_person)
df["sim before ts clean"] = df["knowledge before"].apply(calculate_ts_sim_for_person, args={"no_controlled_successor": True})
df["sim after ts clean"] = df["knowledge after"].apply(calculate_ts_sim_for_person, args={"no_controlled_successor": True})
df["sim before after"] = df[["knowledge before", "knowledge after"]].apply(calculate_sim_between, axis=1)

# Save csv with additional information
df.to_csv("data/knowledge_data_extended.csv", index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")