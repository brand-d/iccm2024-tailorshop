import pandas as pd
import numpy as np
import csv

def rename(x, name):
    return x.replace("action_", name).replace("change_", name)

# Calculates a table where the actions are represented as changes (d1) and changes of changes (d2) instead of the absolute values. Additionally, the knowledge graphs are included.
def get_simplified_df(df, phase="test"):
    results = []
    for idx, row in df.iterrows():
        vp = row["id"]
        if vp not in available_ids:
            continue
        
        if row["phase"] != phase:
            continue
        month = row["month"]
        if month == 1:
            changes_before = {
                rename(x, "d2_"): 0 for x in relevant_change_columns
            }
        knowledge = knowledge_df[knowledge_df["id"] == vp]
        knowledge_before = knowledge["knowledge before"].values
        knowledge_after = knowledge["knowledge after"].values
        success = knowledge["ts success"].values[0]

        starts = {x: row[x] for x in start_columns}
        changes = {
            rename(x, "d1_"): row[x] for x in relevant_change_columns
        }
        
        change_of_changes = {
            rename(x, "d2_"): changes[rename(x, "d1_")] - changes_before[rename(x, "d2_")] for x in relevant_change_columns
        }

        entry = {
            "id": vp,
            "month": row["month"],
            "knowledge_before": knowledge_before,
            "knowledge_after": knowledge_after,
            "success": success,
        }

        entry.update(starts)
        entry.update(changes)
        entry.update(change_of_changes)
        results.append(entry.copy())

        changes_before = {
            rename(x, "d2_"): changes[rename(x, "d1_")] for x in relevant_change_columns
        }
    results_df = pd.DataFrame(results)
    return results_df

# Load the actions table
actions_df = pd.read_csv("data/actions.csv")

# Load the full ts data with knowledge graph information
knowledge_df = pd.read_csv("data/knowledge_data_extended.csv")
knowledge_df = knowledge_df[(knowledge_df["knowledge before"] != "[]") & (knowledge_df["knowledge after"] != "[]")]
available_ids = np.unique(knowledge_df["id"])

relevant_change_columns = [x for x in actions_df.columns \
    if ("change_" in x or "action_" in x) and not "rel" in x]
start_columns = [x for x in actions_df.columns if "start_" in x and "Turn" not in x]


changes_before = {
    rename(x, "d2_"): 0 for x in relevant_change_columns
}

# Generate table for the for the test phase
test_results_df = get_simplified_df(actions_df, phase="test")
test_results_df.to_csv("data/simple_actions_knowledge.csv", index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

# Generate table for the for the exploration phase
exploration_results_df = get_simplified_df(actions_df, phase="exploration")
exploration_results_df.to_csv("data/simple_actions_knowledge_exploration.csv", index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
