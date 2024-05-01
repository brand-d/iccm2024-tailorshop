import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from util import tailorshop

# Move labels away from each other if they are too close
def fix_labels(labels, min_distance=0.1, move_distance=1):
    for i in range(0, len(labels) - 1):
        l1 = labels[i]
        for j in range(i + 1, len(labels)):
            l2 = labels[j]
            l1_pos = np.array(l1.get_position())
            l2_pos = np.array(l2.get_position())
            diff = l1_pos - l2_pos
            distance = np.linalg.norm(diff)
            direction = diff / np.linalg.norm(diff)
            if distance < min_distance:
                l1.set_x(l1_pos[0] + move_distance * direction[0])
                l1.set_y(l1_pos[1] + move_distance * direction[1])
                l2.set_x(l2_pos[0] - move_distance * direction[0])
                l2.set_y(l2_pos[1] - move_distance * direction[1])

# Ignore values too close to zero
def remove_zero_percent(d, precision=1):
    total = 0
    for k, v in d.items():
        total += v
    result = {}
    for k, v in d.items():
        if np.round(v/total * 100, precision) > 0:
            result[k] = v
    return result

# Plot the income pie chart
def plot_income(ax, ts_runs):
    income_cols = [
        'income_Outlets', 
        'income_Machines50', 
        'income_Machines100', 
        'income_Sales', 
        'income_Interest', 
        'income_savings'
    ]
    income_vals = [ts_runs[x].mean() for x in income_cols]
    income_dict = remove_zero_percent(
        dict([(x,y) for x, y in (zip(income_cols, income_vals))]),
        precision=1)
    income_labels = [tailorshop.financeMap[x] for x, y in income_dict.items()] 

    income_cmap = plt.colormaps.get_cmap("Greens")
    income_colors = [income_cmap(x) for x in np.linspace(0.4, 0.6, len(income_dict))]

    # Generate outer circle of pie chart
    wedges, labels, autopct = ax.pie(
            list(income_dict.values()), labels=income_labels,
            pctdistance=0.75, autopct='%.1f%%', labeldistance=1.05,
            radius=1, colors=income_colors, 
            wedgeprops=dict(width=0.5, edgecolor='w'),
            startangle=0)
    fix_labels(autopct, min_distance=0.05, move_distance=1)
    fix_labels(labels, min_distance=0.05, move_distance=1)

    # Generate pie chart inner circle
    centre_circle = plt.Circle((0, 0), 0.6, color=income_cmap(0.65), linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax.text(
        0.5, 0.5, 
        'Income', 
        transform = ax.transAxes, 
        va = 'center', 
        ha = 'center', 
        fontsize=18)

# Plot the expenses pie chart
def plot_expenses(ax, ts_runs, total_invest, total_expense, start_angle=135):
    invest_cols = [x for x in ts_runs.columns if "investments_" in x]
    expense_cols = [x for x in ts_runs.columns if "expenses_" in x]
    
    expenses_cmap = plt.colormaps.get_cmap("Oranges")
    invests_cmap = plt.colormaps.get_cmap("Blues")

    # Generate pie chart inner circle
    wedges, labels, autopct = ax.pie(
            [total_expense, total_invest], labels=["Expense", "Invest"],
            pctdistance=0.3, autopct='%.0f%%', labeldistance=0.7,
            radius=0.6, colors=[expenses_cmap(0.65), invests_cmap(0.6)], 
            wedgeprops=dict(width=0.6, edgecolor='w'),
            startangle=start_angle)
    
    # Rotate the labels correctly
    for i, p in enumerate(labels):
        p.set_ha("center")
        p.set_va("center")
        ang = np.arctan2(p.get_position()[1], p.get_position()[0]) * 180/np.pi
        if ang < 0:
            ang += 360
        ang += 90
        if ang < 180:
            ang += 180
        p.set_rotation(ang) 

    invest_cols = [
        'investments_Machines50', 
        'investments_Machines100', 
        'investments_Outlets'
    ]
    expense_cols = [
        'expenses_Material', 
        'expenses_Outlets', 
        'expenses_Location', 
        'expenses_Storage', 
        'expenses_Advertising', 
        'expenses_Social', 
        'expenses_Salary', 
        'expenses_MachineService'
    ]
    
    # Add interest as an expense in case it was necessary to pay
    if "expenses_Interest" in ts_runs.columns:
        expense_cols.append("expenses_Interest")

    # Remove zero values
    costs_dict = remove_zero_percent(dict( \
            [(x,y) for x, y in (zip(expense_cols, [ts_runs[x].sum() for x in expense_cols]))] \
          + [(x,y) for x, y in (zip(invest_cols, [ts_runs[x].sum() for x in invest_cols]))] \
        ), precision=0)

    # Obtain rows 
    expense_cols = [x for x in expense_cols if x in costs_dict]
    invest_cols = [x for x in invest_cols if x in costs_dict]

    # Create cmap
    expense_colors = [expenses_cmap(x) for x in np.linspace(0.4, 0.6, len(expense_cols))]
    invest_colors = [invests_cmap(x) for x in np.linspace(0.35, 0.5, len(invest_cols))]
    all_colors = expense_colors + invest_colors

    # Prepare labels
    cost_labels = [tailorshop.financeMap[x] for x in costs_dict.keys()]
    
    # Generate outer circle of pie chart
    wedges, labels, autopct = ax.pie(
            list(costs_dict.values()), labels=cost_labels,
            pctdistance=0.8, autopct='%.1f%%', labeldistance=1.1,
            radius=1, colors=all_colors, 
            wedgeprops=dict(width=0.4, edgecolor='w'),
            startangle=start_angle)
    fix_labels(autopct, min_distance=0.15, move_distance=1)
    fix_labels(labels, min_distance=0.15, move_distance=1)

# Calculates the amount either taken from savings 
def get_missing_value(elem):
    missing = 0
    for k, v in elem.items():
        if "income_" in k:
            missing -= v
        else:
            missing += v
    return missing

# Plot the two pie charts
def plot_finance(df, exclude_last=True, title="overall", exp_start_angle=135):
    ts_runs = df.loc[df["phase"] == "test"]
    if exclude_last:
        ts_runs = ts_runs.loc[ts_runs["month"] != 12]

    income_cols = [x for x in ts_runs.columns if "income_" in x]
    invest_cols = [x for x in ts_runs.columns if "investments_" in x]
    expense_cols = [x for x in ts_runs.columns if "expenses_" in x]
    
    # If interest was negative, it was an expense
    ts_runs["expenses_Interest"] = ts_runs["income_Interest"].apply(lambda x: -x if x < 0 else 0)
    expense_cols.append("expenses_Interest")
    
    # Savings/loan is the part of income that is used for expenses not covered by other sources
    ts_runs["income_savings"] = ts_runs[income_cols + invest_cols + expense_cols].apply(get_missing_value, axis=1)
    income_cols.append("income_savings")
    
    # Calculate total invest and expenses
    total_invest = np.sum(ts_runs[invest_cols].sum())
    total_expense = np.sum(ts_runs[expense_cols].sum())

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(8,3))
    plot_expenses(axs[0], ts_runs, total_invest, total_expense, start_angle=exp_start_angle)
    plot_income(axs[1], ts_runs)
    axs[0].set(aspect="equal")
    axs[1].set(aspect="equal")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.2)
    plt.savefig(f"finances_{title}.pdf", bbox_inches='tight', pad_inches = 0)
    plt.show()

# Load the data
ts_df = pd.read_csv("../data/actions.csv")
cm_df = pd.read_csv("../data/knowledge_data_extended.csv")

# Get people with profitable/unprofitable ts
low_cm_df = cm_df[cm_df["ts success"] < 0]
high_cm_df = cm_df[cm_df["ts success"] >= 0]

# Filter by id
high_ts_df = ts_df[ts_df["id"].isin(high_cm_df["id"])]
low_ts_df = ts_df[ts_df["id"].isin(low_cm_df["id"])]

# Plot expenses/investments of profitable TS 
plot_finance(high_ts_df, title="profitable", exp_start_angle=137.5)

# Plot expenses/investments of unprofitable TS
plot_finance(low_ts_df, title="unprofitable", exp_start_angle=147)