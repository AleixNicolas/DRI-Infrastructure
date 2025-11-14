import os
import sys
import math
import json
import networkx as nx
import igraph as ig
import pandas as pd
import geopandas as gpd
import osmnx as ox

def convert_nx_to_igraph(G_nx, weight_attr='travel_time'):
    node_list = list(G_nx.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    edge_weights = {}
    for u, v, attr in G_nx.edges(data=True):
        wt = attr.get(weight_attr)
        if wt is None:
            continue
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        if (u_idx, v_idx) not in edge_weights or wt < edge_weights[(u_idx, v_idx)]:
            edge_weights[(u_idx, v_idx)] = wt

    is_directed = G_nx.is_directed()
    G_ig = ig.Graph(directed=is_directed)
    G_ig.add_vertices(len(node_list))
    G_ig.add_edges(edge_weights.keys())
    G_ig.es['weight'] = list(edge_weights.values())

    return G_ig, node_to_index, index_to_node, edge_weights

def batch_shortest_paths(G_ig, special_nodes, index_to_node, node_to_index, node_to_muni, compute_path, weight_attr='weight'):
    result = {}
    special_indices = [node_to_index[n] for n in special_nodes]
    index_to_muni = {idx: node_to_muni[index_to_node[idx]] for idx in special_indices}

    for src_idx in special_indices:
        distances = G_ig.distances(source=src_idx, target=special_indices, weights=weight_attr)[0]
        paths = None
        if compute_path:
            paths = G_ig.get_shortest_paths(src_idx, to=special_indices, weights=weight_attr, output='vpath')

        for tgt_pos, tgt_idx in enumerate(special_indices):
            if src_idx == tgt_idx:
                continue

            dist = distances[tgt_pos]
            if math.isinf(dist):
                result[f"{index_to_muni[src_idx]}__{index_to_muni[tgt_idx]}"] = ([], 0) if compute_path else 0
                continue

            if compute_path:
                node_path = paths[tgt_pos]
                path = [index_to_node[n] for n in node_path] if node_path else []
                result[f"{index_to_muni[src_idx]}__{index_to_muni[tgt_idx]}"] = (path, dist)
            else:
                result[f"{index_to_muni[src_idx]}__{index_to_muni[tgt_idx]}"] = dist

    return result

def batch_shortest_paths_no_path(G_ig, special_nodes, index_to_node, node_to_index, node_to_muni, weight_attr='weight'):
    result = {}

    special_indices = [node_to_index[n] for n in special_nodes]
    index_to_muni = {idx: node_to_muni[index_to_node[idx]] for idx in special_indices}

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
            key = f"{index_to_muni[src_idx]}__{index_to_muni[tgt_idx]}"
            result[key] = 0 if math.isinf(dist) else dist

    return result

def load_or_compute_shortest_paths(filename, G_nx, special_nodes, node_to_muni, save, compute_path):
    G_ig, node_to_index, index_to_node, edge_weights = convert_nx_to_igraph(G_nx, weight_attr='travel_time')

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        # Deserialize: convert lists to tuples if compute_path is True
        if compute_path:
            result = {k: (v[0], v[1]) for k, v in data.items()}
        else:
            result = data  # just distances
    else:
        print("Computing shortest paths...")
        result = batch_shortest_paths(
            G_ig,
            special_nodes,
            index_to_node,
            node_to_index,
            node_to_muni,
            compute_path=compute_path
        )

        if save:
            if compute_path:
                # Convert tuples to lists for JSON compatibility
                serializable_result = {k: [v[0], v[1]] for k, v in result.items()}
            else:
                # Values are just floats
                serializable_result = result

            with open(filename, 'w') as f:
                json.dump(serializable_result, f)

    return result

def compute_individual_risk_factor(T_P, T_NP):
    if T_P == 0:
        return 1
    else:
        return 1 - (T_NP / T_P)
    
def compute_risk_factor(T_P_dict, T_NP_dict):
    keys = set(T_P_dict.keys()) & set(T_NP_dict.keys())
    R = 0
    for k in keys:
        T_P_time = T_P_dict[k][1]
        T_NP_time = T_NP_dict[k][1]
        R += compute_individual_risk_factor(T_P_time, T_NP_time)
    R /= len(keys) if keys else 1
    return R

def compute_municipal_risk_factor(T_P_dict, T_NP_dict, municipality):
    keys = {k for k in T_P_dict.keys() & T_NP_dict.keys() if municipality in k}
    R = 0
    for k in keys:
        T_P_time = T_P_dict[k][1]
        T_NP_time = T_NP_dict[k][1]
        R += compute_individual_risk_factor(T_P_time, T_NP_time)
    R /= len(keys) if keys else 1
    return R

def tag_flooded_roads(name):
    output_path = cut_roads_files[name]
    graphml_path = safe_roads_files[name]

    print(f"Loading {name} from {output_path}")
    G_safe = ox.load_graphml(graphml_path)
    edges = gpd.read_file(output_path, layer=name)
    if all(col in edges.columns for col in ["u", "v", "key"]):
        edges = edges.set_index(["u", "v", "key"])


    return edges, G_safe


input_dir = "source_files"
output_dir = "processed_files"

ox.settings.use_cache = True
ox.settings.log_console = True

cut_roads_files = {
    "10 yr": f"{output_dir}/cut_roads_flood_risk_10.graphml",
    "100 yr": f"{output_dir}/cut_roads_flood_risk_100.graphml",
    "500 yr": f"{output_dir}/cut_roads_flood_risk_500.graphml",
    "DANA_31_10_2024": f"{output_dir}/cut_roads_DANA_31_10_2024.graphml",
    "DANA_03_11_2024": f"{output_dir}/cut_roads_DANA_03_11_2024.graphml",
    "DANA_05_11_2024": f"{output_dir}/cut_roads_DANA_05_11_2024.graphml",
    "DANA_06_11_2024": f"{output_dir}/cut_roads_DANA_06_11_2024.graphml",
    "DANA_08_11_2024": f"{output_dir}/cut_roads_DANA_08_11_2024.graphml"
}

safe_roads_files = {
    "10 yr": f"{output_dir}/safe_roads_flood_risk_10.graphml",
    "100 yr": f"{output_dir}/safe_roads_flood_risk_100.graphml",
    "500 yr": f"{output_dir}/safe_roads_flood_risk_500.graphml",
    "DANA_31_10_2024": f"{output_dir}/safe_roads_DANA_31_10_2024.graphml",
    "DANA_03_11_2024": f"{output_dir}/safe_roads_DANA_03_11_2024.graphml",
    "DANA_05_11_2024": f"{output_dir}/safe_roads_DANA_05_11_2024.graphml",
    "DANA_06_11_2024": f"{output_dir}/safe_roads_DANA_06_11_2024.graphml",
    "DANA_08_11_2024": f"{output_dir}/safe_roads_DANA_08_11_2024.graphml"
}

shortest_path_files = {
    "Normal Conditions": f"{output_dir}/shorthest_paths_NP.json", 
    "10 yr": f"{output_dir}/shorthest_paths_10.json",
    "100 yr": f"{output_dir}/shorthest_paths_100.json",
    "500 yr": f"{output_dir}/shorthest_paths_500.json",
    "DANA_31_10_2024": f"{output_dir}/shorthest_paths_DANA_31_10_2024.json",
    "DANA_03_11_2024": f"{output_dir}/shorthest_paths_DANA_03_11_2024.json",
    "DANA_05_11_2024": f"{output_dir}/shorthest_paths_DANA_05_11_2024.json",
    "DANA_06_11_2024": f"{output_dir}/shorthest_paths_DANA_06_11_2024.json",
    "DANA_08_11_2024": f"{output_dir}/shorthest_paths_DANA_08_11_2024.json"
}

graph_path = os.path.join(output_dir, "G_2nd.graphml")
G_2nd = ox.load_graphml(graph_path)
print(f"Loaded saved graph from {graph_path}")

filename = shortest_path_files["Normal Conditions"]

special_nodes = [n for n, attr in G_2nd.nodes(data=True) if attr.get('municipality')]
node_to_muni = {n: attr.get('municipality', '') for n, attr in G_2nd.nodes(data=True)}

T_NP_dictionary = load_or_compute_shortest_paths(filename, G_2nd, special_nodes, node_to_muni,True,True)

layer_names=["10 yr","100 yr","500 yr","DANA_31_10_2024","DANA_03_11_2024","DANA_05_11_2024","DANA_06_11_2024","DANA_08_11_2024"]
flood_edges_var = {}
flood_graph_var = {}

for name in layer_names:
    result_1, result_2 = tag_flooded_roads(name)
    flood_edges_var[name] = result_1
    flood_graph_var[name] = result_2

for name, graph in flood_graph_var.items():
    components = list(nx.weakly_connected_components(graph))
    
    multi_muni_count = sum(
        1 for component in components
        if len({graph.nodes[node]["municipality"] for node in component if graph.nodes[node]["municipality"] != ""}) > 0
    )

    print(f"{name} {multi_muni_count} weakly connected components")

    if multi_muni_count > 0:
        j = 0
        for i, component in enumerate(components):
            municipalities = {
                graph.nodes[node]["municipality"]
                for node in component
                if graph.nodes[node]["municipality"] != ""
            }
            if len(municipalities) > 0:
                j += 1
                print(f"Component {j} municipalities: {municipalities}")

layer_names=["10 yr","100 yr","500 yr","DANA_31_10_2024","DANA_03_11_2024","DANA_05_11_2024","DANA_06_11_2024","DANA_08_11_2024"]
#layer_names=["DANA_31_10_2024"]
T_P_dictionaries = {}
R={}

for i, name in enumerate(layer_names):
    T_P_dictionaries[name] = load_or_compute_shortest_paths(shortest_path_files[name], flood_graph_var[name], special_nodes, node_to_muni,True,True)
    R[name] = compute_risk_factor(T_P_dictionaries[name], T_NP_dictionary)
    
    percent_complete = (i + 1) / len(layer_names) * 100
    print(f"\rProgress: {percent_complete:.2f}% ({i + 1}/{len(layer_names)})", end="")

with open("processed_files/R_G.json", "w") as f:
    json.dump(R, f, indent=2)

used_edges = set()

for path, _ in T_NP_dictionary.values():
    for u, v in zip(path, path[1:]):
        used_edges.add((u, v))

G_ig, node_to_index, index_to_node, edge_weights = convert_nx_to_igraph(G_2nd, weight_attr='travel_time')
special_indices = [node_to_index[n] for n in special_nodes]

edge_risks_path = "processed_files/edge_risks_NP.json"

# Check if the file already exists
if os.path.exists(edge_risks_path):
    print("Edge risks already computed")
else:
    print("Computing edge risks")
    base_risk = 0
    edge_risks = []
    i=0
    total = len(used_edges)
    #total = 100

    for u, v in list(used_edges)[:total]:
        i+=1
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        eid = G_ig.get_eid(u_idx, v_idx, directed=True, error=False)
        if eid == -1:
            continue

        G_ig.delete_edges(eid)

        T_P_dictionary = batch_shortest_paths_no_path(G_ig, special_nodes, index_to_node, node_to_index, node_to_muni)

        new_risk = compute_risk_factor_2(T_P_dictionary, T_NP_dictionary)
        delta_risk = new_risk - base_risk

        edge_risks.append(((u, v), delta_risk))

        G_ig.add_edges([(u_idx, v_idx)])
        G_ig.es[-1]['weight'] = edge_weights[(u_idx, v_idx)]

        percent_complete = (i) / total * 100
        print(f"\rProgress: {percent_complete:.2f}% ({i}/{total})", end="")
        sys.stdout.flush()

    edge_risks.sort(key=lambda x: x[1], reverse=True)
    edge_risks_json = [ {"edge": [u, v], "delta_risk": delta_risk} for (u, v), delta_risk in edge_risks ]
    with open("processed_files/edge_risks_NP.json", "w") as f:
        json.dump(edge_risks_json, f, indent=2)

    

if os.path.exists("processed_files/edge_risks_DANA.json"):
    print("Edge risks already computed")
    
else:
    # Step 1: Convert reduced graph to igraph
    G_ig_dana, node_to_index, index_to_node, _ = convert_nx_to_igraph(
    flood_graph_var["DANA_31_10_2024"], weight_attr='travel_time')

    # Step 2: Prepare special node indices
    special_indices = [node_to_index[n] for n in special_nodes]

    # Step 3: Compute baseline risk for reduced graph
    T_P_dictionary = batch_shortest_paths(
        G_ig_dana, special_nodes, index_to_node, node_to_index, node_to_muni, compute_path=False)

    base_risk = compute_risk_factor(
        T_P_dictionaries["DANA_31_10_2024"], T_NP_dictionary)

    # Step 4: Prepare candidate edges (only in flood zone)
    candidate_edges = flood_edges_var["DANA_31_10_2024"]
    candidate_edges = candidate_edges[candidate_edges['in_flood_zone'].astype(bool)].copy()

    edge_risks = []
    total = len(candidate_edges)
    for i, ((u, v, _), row) in enumerate(candidate_edges.iterrows()):
        weight = row['travel_time']

        if u not in node_to_index or v not in node_to_index:
            continue

        u_idx, v_idx = node_to_index[u], node_to_index[v]

        # Add edge
        G_ig_dana.add_edges([(u_idx, v_idx)])
        eid = G_ig_dana.get_eid(u_idx, v_idx, directed=True, error=False)
        if eid == -1:
            continue

        G_ig_dana.es[eid]['weight'] = weight

        # Recompute risk
        T_P_dictionary = batch_shortest_paths_no_path(G_ig_dana, special_nodes, index_to_node, node_to_index, node_to_muni)
        new_risk = compute_risk_factor_2(T_P_dictionary, T_NP_dictionary)
        delta_risk = new_risk - base_risk

        edge_risks.append(((u, v), delta_risk))

        # Remove the edge
        G_ig_dana.delete_edges(eid)

        # Print progress
        percent_complete = (i + 1) / total * 100
        print(f"\rProgress: {percent_complete:.2f}% ({i + 1}/{total})", end="")
        sys.stdout.flush()


    edge_risks.sort(key=lambda x: x[1], reverse=True)

    edge_risks_json = [ {"edge": [u, v], "delta_risk": delta_risk} for (u, v), delta_risk in edge_risks ]
    with open("processed_files/edge_risks_DANA.json", "w") as f:
        json.dump(edge_risks_json, f, indent=2)