import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from util import tailorshop

# Map for convenient renaming of the variables
action_map = {
    'd1_Workers50'         : "W.50",
    'd1_Workers100'        : "W.100",
    'd1_WorkerSalary'      : "Salary",
    'd1_ShirtPrice'        : "Price",
    'd1_SalesOutlets'      : "Outlets",
    'd1_MaterialOrder'     : "Material",
    'd1_Machines50'        : "M.50",
    'd1_Machines100'       : "M.100",
    'd1_MachineService'    : "Repair",
    'd1_WorkerBenefits'    : "Social",
    'd1_Advertising'       : "Advert.",
    'd1_BusinessLocation'  : "Location",
    'd1_BankAccount'       : "Account",
    'd1_ShirtSales'        : "Sales",
    'd1_MaterialPrice'     : "Mat.Price",
    'd1_ShirtStock'        : "Sh.Stock",
    'd1_WorkerSatisfaction': "W.Satis",
    'd1_ProductionIdle'    : "Prod.Idle",
    'd1_CompanyValue'      : "Comp.Val",
    'd1_CustomerInterest'  : "Cust.Int",
    'd1_MaterialStock'     : "Mat.Stock",
    'd1_MachineCapacity'   : "Damage",
}

# Obtain a matrix containing the normalized means per month
def get_mat_by_month(df, columns):
    maxes = df[columns].max().values
    mins = np.abs(df[columns].min().values)
    
    maxes = np.maximum(maxes, mins)
    grp = df.groupby("month")[columns].agg("mean")
    mat = grp.values
    mat /= maxes
    return mat

# Plots a single matrix as a heatmap to a given axis
def plot_heatmap(actions_mat, ax, cols, total_max, title,
                cbar_ax=None, show_yaxis=True,
                yticks=12, ylabel="Month",
                show_x_label=True):

    # Plot the matrix
    sns.heatmap(actions_mat, ax=ax, 
                vmin=-total_max, vmax=total_max, 
                cmap="RdBu", cbar=(cbar_ax is not None), cbar_ax=cbar_ax)

    # Style axes
    if show_yaxis:
        ax.set_ylabel(ylabel, fontsize=12, ha="center", va="center")
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_yticklabels([x + 1 for x in range(yticks)], rotation=0, fontsize=12)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    if show_x_label:
        ax.set_xticklabels([action_map[x] for x in cols], rotation=80, fontsize=11)
    else:
        ax.set_xticklabels([])

    # Set title if available
    if title is not None:
        ax.set_title(title, fontsize=14)

# Plots actions and the corresponding observations into two axes
def plot_situation(df, axs, yticks=12, show_subfig_title=False, ylabel="Month", show_x_label=True):
    action_mat = get_mat_by_month(df, action_cols)
    observ_mat = get_mat_by_month(df, observ_cols)
    act_max = np.max(np.abs(action_mat))
    obs_max = np.max(np.abs(observ_mat))

    act_title = "Actions" if show_subfig_title else None
    obs_title = "Observations" if show_subfig_title else None

    plot_heatmap(action_mat, axs[0], action_cols, act_max, act_title, cbar_ax=None, show_yaxis=True, yticks=yticks, ylabel=ylabel, show_x_label=show_x_label)
    plot_heatmap(observ_mat, axs[1], observ_cols, obs_max, obs_title, cbar_ax=None, show_yaxis=False, yticks=yticks, ylabel=ylabel, show_x_label=show_x_label)

def plot_action_map(test_phase, explo_phase, fname):
    # Prepare figure with 4 subplots
    fig, axs = plt.subplots(2, 2,
        figsize=(6, 4.8), 
        gridspec_kw={'width_ratios': [1, len(observations)/len(actions)],
                    'height_ratios': [0.5, 1]
    })

    # Define axes 
    test_axs = [axs[1, 0], axs[1, 1]]
    explo_axs = [axs[0, 0], axs[0, 1]]

    # Plot for test phase data
    plot_situation(test_phase, test_axs,
                yticks=12,
                ylabel="Month (Test)")

    # Plot for exploration phase data
    plot_situation(explo_phase, explo_axs, 
                yticks=6,
                ylabel="Month (Expl.)",
                show_subfig_title=True,
                show_x_label=False)

    # Finish plot, save and show
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', pad_inches = 0)
    plt.show()

# List of actions to include
actions = [
            'Workers50', 'Workers100', 'WorkerSalary', 'WorkerBenefits', 
            'Machines50', 'Machines100', 'MachineService',
            'SalesOutlets', 'BusinessLocation', 'Advertising',
            'ShirtPrice',  'MaterialOrder'
]

# List of observation variables to include
observations = [
            'BankAccount', 'CompanyValue',
            'CustomerInterest', 'ShirtSales', 'ShirtStock',
            'MaterialStock', 'ProductionIdle',
            'WorkerSatisfaction', 'MachineCapacity', 
            'MaterialPrice'
]

# Create list of columns to load for each action/observation
action_cols = [f"d1_{x}" for x in actions]
observ_cols = [f"d1_{x}" for x in observations]

# Load the test phase dataset
actions_df = pd.read_csv("../data/simple_actions_knowledge.csv")

# Filter columns
relevant_columns = ["month", "id", "success"] + \
                   [x for x in actions_df.columns if "d1_" in x]
actions_df = actions_df[relevant_columns]

# Flip the value for machine capacity, since it is used to reflect damage
actions_df["d1_MachineCapacity"] *= -1

# Separate into participants with profitable and non-profitable ts runs
high_perform_df = actions_df[actions_df["success"] >= 0]
low_perform_df = actions_df[actions_df["success"] < 0]
print(f"Numbers of high/low/total: {len(high_perform_df)/12}/{len(low_perform_df)/12}/{len(actions_df)/12}")

# Load and process exploration phase data
explo_df = pd.read_csv("../data/simple_actions_knowledge_exploration.csv")
explo_df = explo_df[relevant_columns]
explo_df["d1_MachineCapacity"] *= -1

# Divide into groups of the same participants that were good/bad perfoming before
explo_high_perform_df = explo_df[explo_df["id"].isin(
    np.unique(high_perform_df["id"]))]
explo_low_perform_df = explo_df[explo_df["id"].isin(
    np.unique(low_perform_df["id"]))]
print(f"Numbers of high/low/total (explo): {len(explo_high_perform_df)/6}/{len(explo_low_perform_df)/6}/{len(explo_df)/6}")

# Plot high performing group
plot_action_map(low_perform_df,
                explo_low_perform_df,
                "actions_observations_high.pdf")

# Plot low performing group
plot_action_map(high_perform_df,
                explo_high_perform_df,
                "actions_observations_low.pdf")
