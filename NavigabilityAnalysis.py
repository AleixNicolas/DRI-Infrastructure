import os
import sys
import math
import json
import networkx as nx
import igraph as ig
import osmnx as ox
import geopandas as gpd

# ===============================================================
# SETTINGS
# ===============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_files")
INPUT_DIR = os.path.join(SCRIPT_DIR, "source_files")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

COMPUTE_T_P = True
COMPUTE_T_NP = True

# ===============================================================
# FILE PATHS
# ===============================================================
GRAPH_NORMAL_PATH = os.path.join(OUTPUT_DIR, "G_2nd.graphml")

shortest_path_files = {
    "Normal Conditions": f"{OUTPUT_DIR}/shortest_paths_NP.json",
    "10 yr": f"{OUTPUT_DIR}/shortest_paths_10.json",
    "100 yr": f"{OUTPUT_DIR}/shortest_paths_100.json",
    "500 yr": f"{OUTPUT_DIR}/shortest_paths_500.json",
    "DANA_31_10_2024": f"{OUTPUT_DIR}/shortest_paths_DANA_31_10_2024.json",
    "DANA_03_11_2024": f"{OUTPUT_DIR}/shortest_paths_DANA_03_11_2024.json",
    "DANA_05_11_2024": f"{OUTPUT_DIR}/shortest_paths_DANA_05_11_2024.json",
    "DANA_06_11_2024": f"{OUTPUT_DIR}/shortest_paths_DANA_06_11_2024.json",
    "DANA_08_11_2024": f"{OUTPUT_DIR}/shortest_paths_DANA_08_11_2024.json"
}

safe_roads_files = {
    name: f"{OUTPUT_DIR}/safe_roads_{name}.graphml"
    for name in ["10 yr", "100 yr", "500 yr",
                 "DANA_31_10_2024", "DANA_03_11_2024",
                 "DANA_05_11_2024", "DANA_06_11_2024", "DANA_08_11_2024"]
}

# ===============================================================
# GRAPH CONVERSION
# ===============================================================
def convert_nx_to_igraph(G_nx, weight_attr="travel_time"):
    """Convert NetworkX graph to igraph for fast shortest paths."""
    node_list = list(G_nx.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    edge_list = []
    weights = []
    for u, v, data in G_nx.edges(data=True):
        w = data.get(weight_attr)
        if w is None:
            continue
        edge_list.append((node_to_index[u], node_to_index[v]))
        weights.append(w)

    G_ig = ig.Graph(directed=G_nx.is_directed())
    G_ig.add_vertices(len(node_list))
    G_ig.add_edges(edge_list)
    G_ig.es["weight"] = weights

    return G_ig, node_to_index, index_to_node


# ===============================================================
# SHORTEST PATH COMPUTATIONS
# ===============================================================
def compute_paths_with_routes(G_ig, special_nodes, node_to_index, index_to_node, node_to_muni):
    """Return dict { 'A__B': { 'path': [...], 'time': X } }."""
    result = {}
    idx_list = [node_to_index[n] for n in special_nodes]

    for src_idx in idx_list:
        distances = G_ig.distances(src_idx, idx_list, weights="weight")[0]
        routes = G_ig.get_shortest_paths(src_idx, to=idx_list, weights="weight", output="vpath")

        src_m = node_to_muni[index_to_node[src_idx]]

        for tgt_idx, dist, vpath in zip(idx_list, distances, routes):
            if src_idx == tgt_idx:
                continue

            tgt_m = node_to_muni[index_to_node[tgt_idx]]
            key = f"{src_m}__{tgt_m}"

            if math.isinf(dist):
                result[key] = {"path": [], "time": None}
            else:
                result[key] = {"path": [index_to_node[i] for i in vpath], "time": dist}

    return result


def compute_times_only(G_ig, special_nodes, node_to_index, index_to_node, node_to_muni):
    """Return dict {'A__B': time}."""
    result = {}
    idx_list = [node_to_index[n] for n in special_nodes]

    dist_matrix = G_ig.shortest_paths_dijkstra(idx_list, idx_list, weights="weight")

    for i, src_idx in enumerate(idx_list):
        for j, tgt_idx in enumerate(idx_list):
            if src_idx == tgt_idx:
                continue

            src_m = node_to_muni[index_to_node[src_idx]]
            tgt_m = node_to_muni[index_to_node[tgt_idx]]
            key = f"{src_m}__{tgt_m}"

            d = dist_matrix[i][j]
            result[key] = None if math.isinf(d) else d

    return result


# ===============================================================
# LOAD OR COMPUTE
# ===============================================================
def load_or_compute_paths(filename, G_nx, special_nodes, node_to_muni):
    """Load or compute full paths + times."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)

    print(f"\nComputing full paths for {filename} ...")
    G_ig, node_to_index, index_to_node = convert_nx_to_igraph(G_nx)
    result = compute_paths_with_routes(G_ig, special_nodes, node_to_index, index_to_node, node_to_muni)

    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ===============================================================
# RISK FUNCTIONS
# ===============================================================
def individual_risk(TP, TNP):
    """Compute risk per pair."""
    if TP is None or math.isinf(TP):
        return 1
    if TP == 0:
        return 0
    return 1 - (TNP / TP)


def compute_risk(TP_dict, TNP_dict):
    keys = set(TP_dict.keys()) & set(TNP_dict.keys())
    if not keys:
        return 0

    vals = []
    for k in keys:
        TP = TP_dict[k]["time"] if isinstance(TP_dict[k], dict) else TP_dict[k]
        TNP = TNP_dict[k]["time"] if isinstance(TNP_dict[k], dict) else TNP_dict[k]
        if TNP is None:
            continue
        vals.append(individual_risk(TP, TNP))

    return sum(vals) / len(vals) if vals else 0


# ===============================================================
# MAIN
# ===============================================================
def main():
    G_2nd = ox.load_graphml(GRAPH_NORMAL_PATH)
    print(f"Loaded normal graph: {GRAPH_NORMAL_PATH}")

    special_nodes = [n for n, d in G_2nd.nodes(data=True) if d.get("municipality")]
    node_to_muni = {n: d.get("municipality", "") for n, d in G_2nd.nodes(data=True)}

    # ---- Compute T_NP (times only) ----
    if COMPUTE_T_NP:
        T_NP = load_or_compute_paths(shortest_path_files["Normal Conditions"], G_2nd, special_nodes, node_to_muni)
    else:
        T_NP = {}

    # ---- Load flooded safe graphs ----
    scenarios = ["10 yr", "100 yr", "500 yr",
                 "DANA_31_10_2024", "DANA_03_11_2024",
                 "DANA_05_11_2024", "DANA_06_11_2024", "DANA_08_11_2024"]

    TP = {}
    R = {}

    if COMPUTE_T_P:
        for i, name in enumerate(scenarios):
            print(f"\n=== Scenario: {name} ===")
            G_safe = ox.load_graphml(safe_roads_files[name])

            # compute full paths
            TP[name] = load_or_compute_paths(shortest_path_files[name], G_safe, special_nodes, node_to_muni)

            # compute scenario risk
            R[name] = compute_risk(TP[name], T_NP)

            print(f"Progress: {i+1}/{len(scenarios)} → R = {R[name]:.4f}")

        # save risk
        with open(os.path.join(OUTPUT_DIR, "R_G.json"), "w") as f:
            json.dump(R, f, indent=2)

    print("\nAll computations finished.")


if __name__ == "__main__":
    main()
