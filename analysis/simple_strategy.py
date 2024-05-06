import numpy as np
import pandas as pd
from scipy import stats

def strategy1(elem):
    res = (np.sign(elem["d1_Machines100"]) + np.sign(elem["d1_Workers100"])) * -np.sign(elem["d1_Machines50"])
    return res

def strategy2(elem):
    res = elem["d1_MaterialOrder"] + elem["d1_MachineService"] 
    return res

# Load data
df = pd.read_csv("../data/simple_actions_knowledge.csv")
df["profitable"] = df["success"] >= 0

# Only strategies based on the first month are considered
first_month = df[df["month"] == 1].copy()
print("First month strategies: Spearman with final success (ratio of increased company value)")

# Simple strategy 1: Buy machine100, hire worker100, but if so, important to sell machine50
# --> Modernize
first_month["strategy_1"] = \
    first_month[["d1_Machines50", "d1_Machines100", "d1_Workers100",]].apply(strategy1,axis=1)
success_strat1 = stats.spearmanr(first_month["strategy_1"], first_month["success"], alternative="greater")
profitable_strat1 = stats.spearmanr(first_month["strategy_1"], first_month["profitable"], alternative="greater")
print("Strategy 1 [Success]:     r={}\tp={}".format(
    str(np.round(success_strat1.statistic, 3)).ljust(5, " "),
    str(np.round(success_strat1.pvalue, 3)).ljust(5, " ")))
print("Strategy 1 [Profitable]:  r={}\tp={}".format(
    str(np.round(profitable_strat1.statistic, 3)).ljust(5, " "),
    str(np.round(profitable_strat1.pvalue, 3)).ljust(5, " ")))
print()

# Simple strategy 2: Buy much material, increase repair
# --> Maximize production with given things
first_month["strategy_2"] = \
    first_month[["d1_MaterialOrder", "d1_MachineService", "d1_WorkerBenefits", "d1_WorkerSalary"]] \
    .apply(strategy2,axis=1)
success_strat2 = stats.spearmanr(first_month["strategy_2"], first_month["success"], alternative="greater")
profitable_strat2 = stats.spearmanr(first_month["strategy_2"], first_month["profitable"], alternative="greater")
print("Strategy 2 [Success]:     r={}\tp={}".format(
    str(np.round(success_strat2.statistic, 3)).ljust(5, " "),
    str(np.round(success_strat2.pvalue, 3)).ljust(5, " ")))
print("Strategy 2 [Profitable]:  r={}\tp={}".format(
    str(np.round(profitable_strat2.statistic, 3)).ljust(5, " "),
    str(np.round(profitable_strat2.pvalue, 3)).ljust(5, " ")))
print()
