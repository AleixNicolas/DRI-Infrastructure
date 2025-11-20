import os
import math
import json
import networkx as nx
import igraph as ig
import osmnx as ox
import geopandas as gpd


# ===============================================================
# 1. PATH SETUP (uses the current script location)
# ===============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_files")

GRAPH_NORMAL_PATH = os.path.join(OUTPUT_DIR, "G_2nd.graphml")

SAFE_ROADS_DANA = os.path.join(OUTPUT_DIR, "safe_roads_DANA_31_10_2024.graphml")

ox.settings.use_cache = True
ox.settings.log_console = True


# ===============================================================
# Convert NetworkX → igraph
# ===============================================================
def build_igraph(G_nx, weight_attr="travel_time"):
    node_list = list(G_nx.nodes())
    node_to_index = {n: i for i, n in enumerate(node_list)}
    index_to_node = {i: n for n, i in node_to_index.items()}

    edges = []
    weights = []

    for u, v, data in G_nx.edges(data=True):
        if weight_attr not in data:
            continue
        edges.append((node_to_index[u], node_to_index[v]))
        weights.append(data[weight_attr])

    G_ig = ig.Graph(directed=G_nx.is_directed())
    G_ig.add_vertices(len(node_list))
    G_ig.add_edges(edges)
    G_ig.es["weight"] = weights

    return G_ig, node_to_index, index_to_node


# ===============================================================
# Main computation:
# shortest municipality → closest node with each "clase"
# ===============================================================
def shortest_to_clase(G_nx, municipio_attr="municipality", clase_attr="clase"):

    G_ig, node_to_index, index_to_node = build_igraph(G_nx, weight_attr="travel_time")

    muni_nodes = {}
    clase_nodes = {}

    for n, data in G_nx.nodes(data=True):
        muni = data.get(municipio_attr)
        clase = data.get(clase_attr)

        if muni not in (None, "", "None"):
            muni_nodes.setdefault(muni, []).append(n)

        if clase not in (None, "", "None"):
            clase_nodes.setdefault(clase, []).append(n)

    results = {}

    for muni, muni_nlist in muni_nodes.items():

        muni_indices = [node_to_index[n] for n in muni_nlist]
        results[muni] = {}

        for clase_val, clase_nlist in clase_nodes.items():

            clase_indices = [node_to_index[n] for n in clase_nlist]

            dist_matrix = G_ig.shortest_paths_dijkstra(
                source=muni_indices,
                target=clase_indices,
                weights="weight"
            )

            min_dist = math.inf
            for row in dist_matrix:
                for d in row:
                    if d < min_dist:
                        min_dist = d

            results[muni][clase_val] = 0 if math.isinf(min_dist) else min_dist

    return results


# ===============================================================
# 4. Load and run the calculations
# ===============================================================
def main():

    # ===============================================================
    # A) NORMAL SCENARIO
    # ===============================================================
    print(f"Loading normal graph: {GRAPH_NORMAL_PATH}")
    G_normal = ox.load_graphml(GRAPH_NORMAL_PATH)
    print("Graph loaded")

    print("Computing shortest municipality → clase (NORMAL)...")
    results_normal = shortest_to_clase(G_normal)

    out_normal = os.path.join(OUTPUT_DIR, "municipality_clase_shortest_Normal.json")
    with open(out_normal, "w") as f:
        json.dump(results_normal, f, indent=2)
    print(f"Saved NORMAL results to:\n{out_normal}")



    # ===============================================================
    # B) DANA SCENARIO
    # ===============================================================
    print(f"Loading DANA safe graph: {SAFE_ROADS_DANA}")
    G_dana = ox.load_graphml(SAFE_ROADS_DANA)
    print("DANA graph loaded")

    print("Computing shortest municipality → clase (DANA)...")
    results_dana = shortest_to_clase(G_dana)

    out_dana = os.path.join(OUTPUT_DIR, "municipality_clase_shortest_DANA_31_10_2024.json")
    with open(out_dana, "w") as f:
        json.dump(results_dana, f, indent=2)
    print(f"Saved DANA results to:\n{out_dana}")


if __name__ == "__main__":
    main()

