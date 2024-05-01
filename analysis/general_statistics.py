import pandas as pd
import numpy as np
from scipy import stats

def relevance_analysis(df):
    relevances = [x for x in df.columns if "Relevance" in x]
    top_list = []
    for variable in relevances:
        top_list.append((variable, df[variable].mean()))
    top_list = sorted(top_list, key=lambda x: x[1], reverse=True)
    
    print("Relevance of variables:")
    for variable, importance in top_list:
        var_name = variable.replace("Relevance ", "") + ":"
        var_name = var_name.ljust(20, " ")
        print(f"    {var_name} {np.round(importance, 3)}")
    

# Load the data with extended information
df = pd.read_csv("../data/knowledge_data_extended.csv")

# Translate gender
df["gender"] = df["gender"].apply(
    lambda x: x.replace("Divers", "diverse")
               .replace("Weiblich", "female")
               .replace("Männlich", "male"))

# Calculate spearmanr for CRT, NFC, Exploration success and before graph sim
corr_crt = stats.spearmanr(df["CRT correct"], df["ts success"])
corr_nfc = stats.spearmanr(df["NFC"], df["ts success"])
corr_exp = stats.spearmanr(df["ts success exploration"], df["ts success"])
corr_before = stats.spearmanr(df["sim before ts"], df["ts success"])

print("Spearman's r:")
print("CRT - TS performance:                         r={}, p={}".format(
    np.round(corr_crt.statistic, 3), np.round(corr_crt.pvalue, 3)))
print("NFC - TS performance:                         r={}, p={}".format(
    np.round(corr_nfc.statistic, 3), np.round(corr_nfc.pvalue, 3)))
print("Exploration performance - TS performance:     r={}, p={}".format(
    np.round(corr_exp.statistic, 3), np.round(corr_exp.pvalue, 3)))
print("sim(before_graph, ts_graph) - TS performance: r={}, p={}".format(
    np.round(corr_before.statistic, 3), np.round(corr_before.pvalue, 3)))
print()

# Graph Similarity
print("Similarity between knowledge graphs:")
print("Before - After:                              ", np.round(df["sim before after"].mean(), 3))
print("Before - TS Truth:                           ", np.round(df["sim before ts"].mean(), 3))
print("After  - TS Truth:                           ", np.round(df["sim after ts"].mean(), 3))
u, p = stats.mannwhitneyu(df["sim before ts"].values, df["sim after ts"].values)
print("Mann-Whitney-U:                               U={}, p={}".format(u, np.round(p, 3)))

# Other information
print("Other information")
print("Considered to sell everything:", df["considered sell everything"].mean())
print("Dept rate:", df["debt"].mean())
print("Demographics:", dict(zip(*np.unique(df["gender"], return_counts=True))))
print()

# Relevance of variables
print("Stated relevance of variables")
relevance_analysis(df)