ICCM2024 - Tailorshop
=====================
Companion repository for the 2024 article "Predicting Complex Problem Solving Performance in the Tailorshop Scenario".

## Overview

- `analysis`: Contains the analysis and plotting scripts.
- `analysis/util`: Contains utility scripts.
- `analysis/graph.py`: Contains utility functions for handling the graphs in this analysis.
- `analysis/tailorshop.py`: Contains variable definitions and naming-maps for the tailorshop variables.
- `analysis/action_heatmap.py`: Draws the heatmap visualizing the actions and observable variables per month (Figure 4).
- `analysis/simple_strategy.py`: Outputs the correlations of the simple strategieswith the success in the tailorshop.
- `analysis/graph_analysis.py`: Calculates metrics for the graph, mostly covering Table 1 and Table 2.
- `analysis/general_statistics.py`: Calculates various general statistics (e.g., similarity of the graphs to the ground truth, correlation of individual factors with success, etc) and displays descriptive properties of the data from the surveys (e.g., demographical information, debt-rate, relevance of variables).
- `analysis/plot_graphs.py`: Generates circular plots visualizing a tailorshop dependency graph for the causal maps (before and after participants interacted with the tailorshop) as well as the ground truth (Figure 2).
- `analysis/finance_behavior.py`: Plots the financial behavior as pie plots visualizing the expenses, the income and the investments (Figure 3).
- `analysis/models.py`: Performs the model evaluation.
- `data`: Contains the datasets used for the analyses.
- `data/actions.csv`: Contains actions and observable variables for each month and participant.
- `data/knowledge_data.csv`: Contains knowledge graph information and the data obtained by the survey (relevance of variables, CRT, NFC).
- `data/knowledge_data_extended.csv`: Contains knowledge graph information and the data obtained by the survey (relevance of variables, CRT, NFC) extended with additional information (e.g., similarity of the knowledge graph to the ground truth of the tailorshop, success in the tailorshop, etc).
- `data/simple_actions_knowledge.csv`: Contains actions and observable variables of the test phase as starting values per month, changes per month as well as second-order changes (changes to the changes). Additionally, the knowledge graph information is added.
- `data/simple_actions_knowledge_exploration.csv`: Contains actions and observable variables of the exploration phase as starting values per month, changes per month as well as second-order changes (changes to the changes). Additionally, the knowledge graph information is added.
- `create_simplified_matrix.py`: Creates `simple_actions_knowledge.csv` and `simple_actions_knowledge_exploration.csv` from `knowledge_data_extended.csv` and `actions.csv`.
- `extend_graph.py`: Adds the additional information to `knowledge_data.csv` in order to create `knowledge_data_extended.csv`.

## Dependencies

The analysis need the following dependencies in order to run:

- Python 3
    - [pandas](https://pandas.pydata.org)
    - [numpy](https://numpy.org)
    - [matplotlib](https://matplotlib.org/)
    - [seaborn](https://seaborn.pydata.org)
    - [scikit-learn](https://scikit-learn.org/)
    - [networkx](https://networkx.org/)

## Run the scripts

Once all dependencies are installed, each script can run directly without arguments. All data files required are already in the data folder, so that running `create_simplified_matrix.py` and `extend_graph.py` beforehand is not necessary.
In order to run any of the scripts, use the following commands:

```
cd /path/to/repository/
$> python [script].py
```

Information will be directly printed to the terminal, plots will be placed in the folder containing the scripts.

## References

Brand, D., Todorovikj, S., and Ragni, M. (2024). Predicting Complex Problem Solving Performance in the Tailorshop Scenario. In *Proceedings of the 22th International Conference on Cognitive Modeling*.