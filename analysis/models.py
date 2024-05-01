import pandas as pd
import numpy as np
import json
from util import graph
from util import tailorshop
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def get_action_x_y(df):
    X = []
    y = []
    actions = [x for x, y in tailorshop.ts_replacement_map.items() \
        if y in tailorshop.controllable_variables]
    action_cols = [f"d1_{x}" for x in actions]
    
    for _, row in df.iterrows():
        success = row["success"]
        action_vec = row[action_cols].values

        y.append(success)
        X.append(action_vec)
    return np.array(X), np.array(y)

def get_knowledge_x_y(df, include_indiv_factors=False):
    X = []
    y = []
    for _, row in df.iterrows():
        success = row["ts success"]

        graph_info = row["knowledge before"]
        adj_mat = graph.get_adjacency_matrix(
            graph.get_graph_from_nodeslist( \
                json.loads(graph_info),
                replacement=True)
        )
        val_for_feats = adj_mat.reshape(-1)
        
        if include_indiv_factors:
            features = ["CRT correct", "NFC"]
            additional_feats = np.array(row[features].values, dtype=float)
            val_for_feats = np.append(val_for_feats, additional_feats)

        y.append(success)
        X.append(val_for_feats)
    return np.array(X), np.array(y)

def get_svr_predictions(X, y):
    res = []
    # Perform a leave-one-out cross-validation
    for i in range(len(X)):
        # get test X and y
        test_X = np.array([X[i]])
        test_y = y[i]
        
        # delete test from training
        train_X = np.delete(X, i, axis=0)
        train_y = np.delete(y, i)
        
        # train model
        regr = make_pipeline(
            StandardScaler(), 
            SVR(C=1.0, epsilon=0.1)
        )
        regr.fit(train_X, train_y)
        
        # obtain predicition
        pred_y = regr.predict(test_X.reshape(1,-1))
        res.append(pred_y[0])
    return res

def get_mean_model_predictions(X, y):
    m = np.mean(y)
    return [m] * len(y)

def get_median_model_predictions(X, y):
    med = np.median(y)
    return [med] * len(y)

def rmse(A, B):
    return np.sqrt(mean_squared_error(A, B))

def perform_model_comparison(X, y):
    # Model predictions
    svr_pred = get_svr_predictions(X, y)
    mean_pred = get_mean_model_predictions(X, y)
    median_pred = get_median_model_predictions(X, y)

    # RMSE, MAE and R2
    print("SVR:")
    print("    RMSE:", np.round(rmse(y, svr_pred), 3))
    print("    MAE:", np.round(mean_absolute_error(y, svr_pred), 3))
    print("    R2:", np.round(r2_score(y, svr_pred), 3))
    print()

    print("Mean Model:")
    print("    RMSE:", np.round(rmse(y, mean_pred), 3))
    print("    MAE:", np.round(mean_absolute_error(y, mean_pred), 3))
    print("    R2:", np.round(r2_score(y, mean_pred), 3))
    print()

    print("Median Model:")
    print("    RMSE:", np.round(rmse(y, median_pred), 3))
    print("    MAE:", np.round(mean_absolute_error(y, median_pred), 3))
    print("    R2:", np.round(r2_score(y, median_pred), 3))
    print()

# Load knowledge and remove participants with missing information
df_graph = pd.read_csv("../data/knowledge_data_extended.csv")
df_graph = df_graph[(df_graph["knowledge before"] != "[]") & (df_graph["knowledge after"] != "[]")]

# Obtain data with target=success and input=knowledge graph matrix
X_graph, y_graph = get_knowledge_x_y(df_graph)

# Model evaluation for the knowledge graph data
print("-----------------------Model Evaluation: Knowledge Graph-----------------------")
perform_model_comparison(X_graph, y_graph)

# Load action data 
df_action = pd.read_csv("../data/simple_actions_knowledge.csv")

# Only use the first month
df_action = df_action[df_action["month"] == 1]

# Obtain data with target=success and input=first month actions
X_action, y_action = get_action_x_y(df_action)

# Model evaluation for the first month actions data
print("---------------------Model Evaluation: First Month Actions---------------------")
perform_model_comparison(X_action, y_action)
