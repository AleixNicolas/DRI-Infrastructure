import os
import math
import json
import networkx as nx
import igraph as ig
import osmnx as ox
import pandas as pd
import geopandas as gpd

# -----------------------
# Base directory (script location)
# -----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def rel_path(*paths):
    """Return an absolute path relative to the script location."""
    return os.path.join(SCRIPT_DIR, *paths)

# -----------------------
# Helper Functions
# -----------------------
def convert_nx_to_igraph(G_nx, weight_attr='travel_time'):
    """Convert NetworkX graph to igraph for fast shortest paths."""
    node_list = list(G_nx.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    edge_weights = {}
    for u, v, attr in G_nx.edges(data=True):
        wt = attr.get(weight_attr)
        if wt is None:
            continue
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        # If multiple edges exist, take the minimum weight
        if (u_idx, v_idx) not in edge_weights or wt < edge_weights[(u_idx, v_idx)]:
            edge_weights[(u_idx, v_idx)] = wt

    G_ig = ig.Graph(directed=G_nx.is_directed())
    G_ig.add_vertices(len(node_list))
    G_ig.add_edges(edge_weights.keys())
    G_ig.es['weight'] = list(edge_weights.values())
    return G_ig, node_to_index, index_to_node, edge_weights

def batch_shortest_paths_no_path(G_ig, special_nodes, index_to_node, node_to_index, node_to_muni, weight_attr='weight'):
    """Compute shortest path times only between special nodes."""
    result = {}
    special_indices = [node_to_index[n] for n in special_nodes]
    index_to_muni_map = {idx: node_to_muni[index_to_node[idx]] for idx in special_indices}
    dist_matrix = G_ig.shortest_paths_dijkstra(
        source=special_indices,
        target=special_indices,
        weights=weight_attr
    )
    for i, src_idx in enumerate(special_indices):
        for j, tgt_idx in enumerate(special_indices):
            if src_idx == tgt_idx:
                continue
            dist = dist_matrix[i][j]
            key = f"{index_to_muni_map[src_idx]}__{index_to_muni_map[tgt_idx]}"
            result[key] = None if math.isinf(dist) else dist
    return result

def compute_individual_risk_factor(T_P, T_NP):
    """Compute risk for one origin-destination pair."""
    if T_P is None or math.isinf(T_P):
        return 1.0
    if T_P == 0:
        return 0.0
    if T_NP is None:
        return 1.0
    return 1 - (T_NP / T_P)

def compute_risk_factor(T_P_dict, T_NP_dict):
    """Compute average risk across all pairs."""
    keys = set(T_P_dict.keys()) & set(T_NP_dict.keys())
    if not keys:
        return 0.0
    vals = []
    for k in keys:
        T_P_time = T_P_dict[k]['time'] if isinstance(T_P_dict[k], dict) else T_P_dict[k]
        T_NP_time = T_NP_dict[k]['time'] if isinstance(T_NP_dict[k], dict) else T_NP_dict[k]
        vals.append(compute_individual_risk_factor(T_P_time, T_NP_time))
    return sum(vals) / len(vals) if vals else 0.0

def tag_flooded_roads(name, cut_roads_files, safe_roads_files):
    """Load flooded road info and safe road graph."""
    output_path = cut_roads_files[name]
    graphml_path = safe_roads_files[name]
    G_safe = ox.load_graphml(graphml_path)
    edges = gpd.read_file(output_path, layer=name)
    if all(col in edges.columns for col in ["u", "v", "key"]):
        edges = edges.set_index(["u", "v", "key"])
    return edges, G_safe

# -----------------------
# Paths
# -----------------------
output_dir = rel_path("processed_files")
graph_path = os.path.join(output_dir, "G_2nd.graphml")

shortest_path_files = {
    "Normal Conditions": os.path.join(output_dir, "shortest_paths_NP.json"), 
    "10 yr": os.path.join(output_dir, "shortest_paths_10.json"),
    "100 yr": os.path.join(output_dir, "shortest_paths_100.json"),
    "500 yr": os.path.join(output_dir, "shortest_paths_500.json"),
    "DANA_31_10_2024": os.path.join(output_dir, "shortest_paths_DANA_31_10_2024.json"),
    "DANA_03_11_2024": os.path.join(output_dir, "shortest_paths_DANA_03_11_2024.json"),
    "DANA_05_11_2024": os.path.join(output_dir, "shortest_paths_DANA_05_11_2024.json"),
    "DANA_06_11_2024": os.path.join(output_dir, "shortest_paths_DANA_06_11_2024.json"),
    "DANA_08_11_2024": os.path.join(output_dir, "shortest_paths_DANA_08_11_2024.json")
}

cut_roads_files = {
    "DANA_31_10_2024": os.path.join(output_dir, "cut_roads_DANA_31_10_2024.gpkg"),
}

safe_roads_files = {
    "DANA_31_10_2024": os.path.join(output_dir, "safe_roads_DANA_31_10_2024.graphml"),
}

# -----------------------
# Load main graph
# -----------------------
G_2nd = ox.load_graphml(graph_path)
special_nodes = [n for n, attr in G_2nd.nodes(data=True) if attr.get('municipality')]
node_to_muni = {n: attr.get('municipality', '') for n, attr in G_2nd.nodes(data=True)}

# -----------------------
# Load baseline shortest paths (Normal Conditions)
# -----------------------
with open(shortest_path_files["Normal Conditions"], "r") as f:
    T_NP_dictionary = json.load(f)
    T_NP_dictionary = {k: {'path': v.get('path', []), 'time': v.get('time')} for k, v in T_NP_dictionary.items()}

# -----------------------
# Compute risk R for each layer
# -----------------------
layer_names = ["10 yr","100 yr","500 yr","DANA_31_10_2024","DANA_03_11_2024","DANA_05_11_2024","DANA_06_11_2024","DANA_08_11_2024"]
T_P_dictionaries = {}
R = {}

for name in layer_names:
    with open(shortest_path_files[name], "r") as f:
        T_P_dictionaries[name] = json.load(f)
        T_P_dictionaries[name] = {k: {'path': v.get('path', []), 'time': v.get('time')} for k, v in T_P_dictionaries[name].items()}
    R[name] = compute_risk_factor(T_P_dictionaries[name], T_NP_dictionary)
    print(f"{name} risk R: {R[name]:.4f}")

# Save risk R
with open(os.path.join(output_dir, "R_G.json"), "w") as f:
    json.dump(R, f, indent=2)

# -----------------------
# Compute edge risks for Normal Conditions
# -----------------------

NC=False
if NC:
    G_ig, node_to_index, index_to_node, edge_weights = convert_nx_to_igraph(G_2nd)
    used_edges = set()
    for path_dict in T_NP_dictionary.values():
        path = path_dict['path']
        for u, v in zip(path, path[1:]):
            used_edges.add((u, v))

    edge_risks = []
    base_risk = 0.0

    for i, (u, v) in enumerate(used_edges):
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        eid = G_ig.get_eid(u_idx, v_idx, directed=True, error=False)
        if eid == -1:
            continue
        G_ig.delete_edges(eid)
        T_P_temp = batch_shortest_paths_no_path(G_ig, special_nodes, index_to_node, node_to_index, node_to_muni,'weight')
        new_risk = compute_risk_factor(T_P_temp, T_NP_dictionary)
        delta_risk = new_risk - base_risk
        edge_risks.append(((u, v), delta_risk))
        # Re-add edge safely
        G_ig.add_edges([(u_idx, v_idx)])
        new_eid = G_ig.get_eid(u_idx, v_idx, directed=True)
        G_ig.es[new_eid]['weight'] = edge_weights[(u_idx, v_idx)]
        if i % 50 == 0:
            print(f"\rProgress NP: {i}/{len(used_edges)} edges", end="", flush=True)

    edge_risks.sort(key=lambda x: x[1], reverse=True)
    edge_risks_json = [{"edge": [u, v], "delta_risk": delta_risk} for (u, v), delta_risk in edge_risks]

    with open(os.path.join(output_dir, "edge_risks_NP.json"), "w") as f:
        json.dump(edge_risks_json, f, indent=2)

# -----------------------
# Compute edge risks for DANA flood
# -----------------------
DANA=True
if DANA:
    flood_edges, G_dana = tag_flooded_roads("DANA_31_10_2024", cut_roads_files, safe_roads_files)
    G_ig_dana, node_to_index, index_to_node, edge_weights = convert_nx_to_igraph(G_dana)
    candidate_edges = flood_edges[flood_edges['in_flood_zone'].astype(bool)].copy()
    edge_risks_dana = []

    base_risk_dana = compute_risk_factor(T_P_dictionaries["DANA_31_10_2024"], T_NP_dictionary)

    for i, ((u, v, _), row) in enumerate(candidate_edges.iterrows()):
        weight = row['travel_time']
        if u not in node_to_index or v not in node_to_index:
            continue
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        # Add temporary edge
        G_ig_dana.add_edges([(u_idx, v_idx)])
        new_eid = G_ig_dana.get_eid(u_idx, v_idx, directed=True)
        G_ig_dana.es[new_eid]['weight'] = weight
        T_P_temp = batch_shortest_paths_no_path(G_ig_dana, special_nodes, index_to_node, node_to_index, node_to_muni, 'weight')
        new_risk = compute_risk_factor(T_P_temp, T_NP_dictionary)
        delta_risk = new_risk - base_risk_dana
        edge_risks_dana.append(((u, v), delta_risk))
        # Remove temporary edge
        G_ig_dana.delete_edges(new_eid)
        if i % 50 == 0:
            print(f"\rProgress DANA: {i}/{len(candidate_edges)} edges", end="", flush=True)

    edge_risks_dana.sort(key=lambda x: x[1], reverse=True)
    edge_risks_dana_json = [{"edge": [u, v], "delta_risk": delta_risk} for (u, v), delta_risk in edge_risks_dana]

    with open(os.path.join(output_dir, "edge_risks_DANA.json"), "w") as f:
        json.dump(edge_risks_dana_json, f, indent=2)

    print("\nEdge risks computation done for DANA.")
