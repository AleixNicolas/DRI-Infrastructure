import os
import json
import logging
from itertools import combinations
import sys

import math

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import shapely
from shapely.geometry import Point
from shapely.ops import unary_union, nearest_points
from shapely.geometry import LineString

import networkx as nx
import osmnx as ox

import scipy
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

import folium
from folium import GeoJson, LayerControl
from branca.colormap import linear
import branca.colormap as bcm

import igraph as ig
import concurrent.futures

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_geojson_safe(gdf):
    gdf = gdf.copy()
    dt_cols = gdf.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    gdf[dt_cols] = gdf[dt_cols].astype(str)
    for col in gdf.columns:
        if col != "geometry" and not pd.api.types.is_scalar(gdf[col].iloc[0]):
            gdf.drop(columns=[col], inplace=True)
    return gdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#STREET DATA

def download_municipalities(names, region="Valencia, Spain"):
    if not names:  # if no specific names provided, download all in region
        try:
            region_gdf = ox.geocode_to_gdf(region)
            polygon = region_gdf.geometry.iloc[0]

            # Download all admin_level=8 (municipalities) overlapping polygon
            gdf = ox.geometries_from_polygon(polygon, tags={"boundary": "administrative"})
            
            # Explicitly filter to admin_level=8 only
            gdf = gdf[gdf["admin_level"] == "8"]
            
            # Continue as before
            gdf = gdf[["geometry", "name"]].reset_index(drop=True)
            gdf = gdf[gdf["name"].notna()]
            gdf_clipped = gpd.clip(gdf, polygon)
            gdf_clipped = gdf_clipped[~gdf_clipped.is_empty]
            gdf_clipped = gdf_clipped.reset_index(drop=True)
            gdf_clipped["name"] = gdf_clipped["name"].astype(str)
            
            return gdf_clipped
        except Exception as e:
            raise RuntimeError(f"Failed to download all municipalities in {region}: {e}")
    else:
        gdfs = []
        for name in names:
            query = f"{name}, {region}"
            try:
                gdf = ox.geocode_to_gdf(query)
                gdf["name"] = name
                gdfs.append(gdf)
            except Exception as e:
                print(f"Failed to download {query}: {e}")
        if not gdfs:
            raise ValueError("No municipalities could be downloaded.")
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    
    from shapely.geometry import MultiPolygon
def build_graph_from_layer(layer_gdf, graph_path, urban_center_dict, network_type="drive"):
    if os.path.exists(graph_path):
        logging.info(f"Loading saved graph from {graph_path}...")
        G = ox.load_graphml(graph_path)
    else:
        logging.info(f"Building graph for {graph_path}...")
        polygon = layer_gdf.unary_union
        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda a: a.area)

        G = ox.graph_from_polygon(
            polygon,
            network_type=network_type,
            simplify=True,
            retain_all=False,
            truncate_by_edge=True
        )

        # Estimate speed & travel time
        for u, v, k, data in G.edges(keys=True, data=True):
            if "length" in data:
                speed = None
                maxspeed = data.get("maxspeed")
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                if maxspeed:
                    try:
                        speed = float(str(maxspeed).split()[0])
                    except ValueError:
                        pass

                if speed is None:
                    highway = data.get("highway")
                    if isinstance(highway, list):
                        highway = highway[0]
                    speed = {
                        "motorway": 120, "motorway_link": 60, "trunk": 100,
                        "primary": 80, "secondary": 60, "tertiary": 50,
                        "residential": 30, "living_street": 10,
                        "unclassified": 40, "service": 20
                    }.get(highway, 50)

                surface = data.get("surface", "").lower()
                factor_map = {
                    "paved": 1.0, "asphalt": 1.0, "concrete": 1.0,
                    "cobblestone": 0.8, "gravel": 0.7, "dirt": 0.6,
                    "ground": 0.6, "sand": 0.5, "unpaved": 0.7,
                    "compacted": 0.85, "fine_gravel": 0.9
                }

                for key, factor in factor_map.items():
                    if key in surface:
                        speed *= factor
                        break

                speed_mps = speed * 1000 / 3600
                data["travel_time"] = data["length"] / speed_mps + 5  # Add turn penalty

        ox.save_graphml(G, graph_path)
        logging.info(f"Graph saved to {graph_path}")

    # Load node GeoDataFrame
    nodes, _ = ox.graph_to_gdfs(G)
    
    # Check if municipality info already exists
    if "municipality" not in nodes.columns or nodes["municipality"].replace("", np.nan).isna().all():
        logging.info("Assigning municipality names to graph nodes...")
        node_coords = np.array([(geom.y, geom.x) for geom in nodes.geometry])
        kdtree = cKDTree(node_coords)

        if "municipality" not in nodes.columns:
            nodes["municipality"] = ""

        for name, (lat, lon) in urban_center_dict.items():
            try:
                _, idx = kdtree.query([lat, lon], k=1)
                nearest_node = nodes.index[idx]
                nodes.at[nearest_node, "municipality"] = name
            except Exception as e:
                logging.warning(f"Could not assign node for {name}: {e}")

        # Update graph with municipality info
        for node_id, row in nodes.iterrows():
            G.nodes[node_id]["municipality"] = row["municipality"]

        ox.save_graphml(G, graph_path)
        logging.info(f"Updated graph saved to: {graph_path}")
    else:
        logging.info("Municipality names already exist in graph; skipping reassignment.")

    return G

def assign_attributes_from_gdf(G, gdf, attr1, attr2, graph_path=None):
    # Load nodes as GeoDataFrame
    nodes, _ = ox.graph_to_gdfs(G)

    # Prepare KDTree of node coordinates
    node_coords = np.array([(geom.y, geom.x) for geom in nodes.geometry])
    kdtree = cKDTree(node_coords)

    # Initialize new attributes if not present
    if attr1 not in nodes.columns:
        nodes[attr1] = None
    if attr2 not in nodes.columns:
        nodes[attr2] = None

    # Assign attributes from closest node
    for idx, row in gdf.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        try:
            _, nearest_idx = kdtree.query([lat, lon], k=1)
            nearest_node = nodes.index[nearest_idx]
            nodes.at[nearest_node, attr1] = row[attr1]
            nodes.at[nearest_node, attr2] = row[attr2]
        except Exception as e:
            logging.warning(f"Could not assign node for point {idx}: {e}")

    # Update graph nodes
    for node_id, row in nodes.iterrows():
        G.nodes[node_id][attr1] = row[attr1]
        G.nodes[node_id][attr2] = row[attr2]

    # Save graph if path provided
    if graph_path:
        ox.save_graphml(G, graph_path)
        logging.info(f"Updated graph saved to: {graph_path}")

    return G

output_dir = "processed_files"
os.makedirs(output_dir, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

FILE_STUDY = os.path.join(output_dir, "affected_area.gpkg")
FILE_1ST = os.path.join(output_dir, "neighbors_1_area.gpkg")
FILE_2ND = os.path.join(output_dir, "neighbors_2_area.gpkg")

polygon_path = os.path.join(output_dir, "study_area.geojson")
graph_path = os.path.join(output_dir, "road_graph.graphml")

with open("source_files/affected_municipalities_dictionary.json", "r", encoding="utf-8") as f:
    regions = json.load(f)
affected_valencia = regions["Valencia"]["affected"]
affected_cuenca = regions["Cuenca"]["affected"]
urban_center = {**regions["Valencia"]["coordinates"], **regions["Cuenca"]["coordinates"]}
del regions

if os.path.exists(FILE_STUDY):
    affected_area = gpd.read_file(FILE_STUDY)
    print(f"Loaded study area from {FILE_STUDY}")
else:
    polygons = []
    valencia_affected_area = download_municipalities(affected_valencia, "Valencia, Spain")
    cuenca_affected_area = download_municipalities(affected_cuenca, "Cuenca, Spain")
    affected_area = gpd.GeoDataFrame(pd.concat([valencia_affected_area, cuenca_affected_area], ignore_index=True))
    del valencia_affected_area, cuenca_affected_area
    affected_area.to_file(FILE_STUDY, driver="GPKG")
    print(f"Saved study area to {FILE_STUDY}")

regions=['Provincia de Valencia, Comunidad Valenciana, Spain', 'Provincia de Castellon, Comunidad Valenciana, Spain', 'Provincia de Cuenca, Castilla-La Mancha, Spain', 'Provincia de Albacete, Castilla-La Mancha, Spain', 'Provincia de Teruel, Aragon, Spain']

if os.path.exists(FILE_1ST):
    neighbors_1_area = gpd.read_file(FILE_1ST)
    print(f"Loaded study area from {FILE_1ST}")
    neighbors_2_area = gpd.read_file(FILE_2ND)
    print(f"Loaded study area from {FILE_2ND}")
else:
    # Concatenate all regions
    possible_neighbors = gpd.GeoDataFrame()
    for region in regions:
        aux = download_municipalities("", region)
        possible_neighbors = pd.concat([possible_neighbors, aux], ignore_index=True)

    # Keep only polygons
    possible_neighbors = possible_neighbors[
        possible_neighbors.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    possible_neighbors = possible_neighbors[possible_neighbors.is_valid & ~possible_neighbors.is_empty]

    possible_neighbors = possible_neighbors.to_crs(affected_area.crs)

    # First ring of neighbors
    union = affected_area.geometry.union_all()
    adjacent_municipalities = possible_neighbors[possible_neighbors.geometry.touches(union)]
    neighbors_1_area = gpd.GeoDataFrame(
        pd.concat([affected_area, adjacent_municipalities], ignore_index=True),
        crs=affected_area.crs
    )
    neighbors_1_area.to_file(FILE_1ST, driver="GPKG")

    # Second ring
    union = neighbors_1_area.geometry.union_all()
    adjacent_municipalities = possible_neighbors[possible_neighbors.geometry.touches(union)]
    neighbors_2_area = gpd.GeoDataFrame(
        pd.concat([neighbors_1_area, adjacent_municipalities], ignore_index=True),
        crs=affected_area.crs
    )
    neighbors_2_area.to_file(FILE_2ND, driver="GPKG")

    print(f"Saved study area to {FILE_2ND}")
    del possible_neighbors, union, adjacent_municipalities

edge_risks_path = "processed_files/points_in_polygon.gpkg"

if os.path.exists(edge_risks_path):
    special_interest_points = gpd.read_file(edge_risks_path)
else:
    special_interest_points_path = (
        "source_files/riesgo-puntos-ambiental-pb-t500_tcm30-176187/"
        "Riesgo_MA_T500_PB_20220627.shp"
    )

    special_interest_points = gpd.read_file(special_interest_points_path)
    special_interest_points = special_interest_points.to_crs(neighbors_2_area.crs)
    special_interest_points = gpd.sjoin(
        special_interest_points, neighbors_2_area, predicate="within"
    )

    keep_cols = ['TIPO_ELTO', 'SUBTIPO_EL', 'geometry']
    special_interest_points = special_interest_points[keep_cols]

    # Save to file
    special_interest_points.to_file(edge_risks_path, driver="GPKG")
    del keep_cols

print("Special interest points imported")

G_affected = build_graph_from_layer(affected_area, os.path.join(output_dir, "G_affected.graphml"), urban_center)
del G_affected
G_1st = build_graph_from_layer(neighbors_1_area, os.path.join(output_dir, "G_1st.graphml"), urban_center)
del G_1st
G_2nd = build_graph_from_layer(neighbors_2_area, os.path.join(output_dir, "G_2nd.graphml"), urban_center)
G_2nd = assign_attributes_from_gdf(G_2nd, special_interest_points, attr1="TIPO_ELTO", attr2="SUBTIPO_EL", graph_path=os.path.join(output_dir, "G_2nd.graphml"))

# FLOOD ZONES

def tag_flooded_roads(edges, nodes, flood_zones, name):
    output_path = cut_roads_files[name]
    graphml_path = safe_roads_files[name]

    #if os.path.exists(output_path) and layer in fiona.listlayers(output_path):
    if os.path.exists(output_path):
        print(f"Loading {name} from {output_path}")
        G_safe = ox.load_graphml(graphml_path)
        edges = gpd.read_file(output_path, layer=name)
        
    else:
        print(f"Tagging and saving {name} to {output_path}")

        bounds = edges.total_bounds
        flood_subset = flood_zones.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        flood_geoms = flood_subset.geometry

        edges = edges.copy()
        edges["in_flood_zone"] = edges.geometry.apply(lambda geom: flood_geoms.intersects(geom).any())

        edges.to_file(output_path, layer=name, driver="GPKG")

    if os.path.exists(graphml_path):
        print("Pruned graph already exists")
    else:    
        safe_edges = edges[~edges["in_flood_zone"]].copy()
        print("Rebuilding pruned graph...")
        G_safe = ox.graph_from_gdfs(nodes, safe_edges)
        ox.save_graphml(G_safe, filepath=graphml_path)
        print(f"Saved pruned graph to {graphml_path}")

    return edges, G_safe

def clip_flood_zone(return_crs, name, clip_geom):
    output_path = zone_output_files[name]
    input_path = zone_input_files[name]
    
    if os.path.exists(output_path):
        print(f"Loading {name} from {output_path}")
        clipped = gpd.read_file(output_path, layer=name).to_crs(return_crs)
    else:
        print(f"Clipping and saving {name} from {output_path}" )
        flood = gpd.read_file(input_path).to_crs(return_crs)
        clipped = gpd.clip(flood, clip_geom)
        clipped.to_file(output_path, layer=name, driver="GPKG")

    return clipped

def parse_depth_range(val):
    if pd.isna(val):
        return None

    val = val.strip()

    if val.startswith('Below'):
        return float(val[5:].strip()) / 2

    if val.startswith('>'):
        return float(val[1:].strip())  # You may want to cap it

    if '-' in val:
        parts = val.split('-')
        try:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        except:
            return None

    try:
        return float(val)
    except:
        return None
    
    def flood_depth_zones(name):
    layer="depth_val"
    input_path = depth_input_files[name]
    output_path=depth_output_files[name]
    if os.path.exists(output_path):
        print(f"Loading {layer} from {output_path}")
        depth=gpd.read_file(output_path, layer=layer)
    else:
        print(f"Saving {layer} to {output_path}")
        depth = gpd.read_file(input_path)
        depth["depth_val"] = depth["value"].apply(parse_depth_range)
        depth.to_file(output_path, layer=layer, driver="GPKG")
        print(f"Saved processed {layer} in {output_path}")
    return depth

input_dir = "source_files"
output_dir = "processed_files"
os.makedirs(output_dir, exist_ok=True)

zone_input_files = {
    "10 yr": f"{input_dir}/laminaspb-q10/Q10_2Ciclo_PB_20241121.shp",
    "100 yr": f"{input_dir}/laminaspb-q100/Q100_2Ciclo_PB_20241121_ETRS89.shp",
    "500 yr": f"{input_dir}/laminaspb-q500/Q500_2Ciclo_PB_20241121_ETRS89.shp",
    "DANA_31_10_2024": f"{input_dir}/EMSR773_AOI01_DEL_PRODUCT_v1/EMSR773_AOI01_DEL_PRODUCT_observedEventA_v1.shp",
    "DANA_03_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT01_v1/EMSR773_AOI01_DEL_MONIT01_observedEventA_v1.shp",
    "DANA_05_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT02_v1/EMSR773_AOI01_DEL_MONIT02_observedEventA_v1.shp",
    "DANA_06_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT03_v1/EMSR773_AOI01_DEL_MONIT03_observedEventA_v1.shp",
    "DANA_08_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT04_v1/EMSR773_AOI01_DEL_MONIT04_observedEventA_v1.shp"
}

depth_input_files = {
    "DANA_31_10_2024": f"{input_dir}/EMSR773_AOI01_DEL_PRODUCT_v1/EMSR773_AOI01_DEL_PRODUCT_floodDepthA_v1.shp",
    "DANA_03_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT01_v1/EMSR773_AOI01_DEL_MONIT01_floodDepthA_v1.shp",
    "DANA_05_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT02_v1/EMSR773_AOI01_DEL_MONIT02_floodDepthA_v1.shp",
    "DANA_06_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT03_v1/EMSR773_AOI01_DEL_MONIT03_floodDepthA_v1.shp",
    "DANA_08_11_2024": f"{input_dir}/EMSR773_AOI01_DEL_MONIT04_v1/EMSR773_AOI01_DEL_MONIT04_floodDepthA_v1.shp"
}

zone_output_files = {
    "10 yr": f"{output_dir}/zone_flood_risk_10.gpkg",
    "100 yr": f"{output_dir}/zone_flood_risk_100.gpkg",
    "500 yr": f"{output_dir}/zone_flood_risk_500.gpkg",
    "DANA_31_10_2024": f"{output_dir}/zone_DANA_31_10_2024.gpkg",
    "DANA_03_11_2024": f"{output_dir}/zone_DANA_03_11_2024.gpkg",
    "DANA_05_11_2024": f"{output_dir}/zone_DANA_05_11_2024.gpkg",
    "DANA_06_11_2024": f"{output_dir}/zone_DANA_06_11_2024.gpkg",
    "DANA_08_11_2024": f"{output_dir}/zone_DANA_08_11_2024.gpkg"
}

depth_output_files = {
    "DANA_31_10_2024": f"{output_dir}/depth_DANA_31_10_2024.gpkg",
    "DANA_03_11_2024": f"{output_dir}/depth_DANA_03_11_2024.gpkg",
    "DANA_05_11_2024": f"{output_dir}/depth_DANA_05_11_2024.gpkg",
    "DANA_06_11_2024": f"{output_dir}/depth_DANA_06_11_2024.gpkg",
    "DANA_08_11_2024": f"{output_dir}/depth_DANA_08_11_2024.gpkg"
}

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

nodes, edges = ox.graph_to_gdfs(G_2nd)

layer_names=["10 yr","100 yr","500 yr","DANA_31_10_2024","DANA_03_11_2024","DANA_05_11_2024","DANA_06_11_2024","DANA_08_11_2024"]
#layer_names=["DANA_31_10_2024"]
flood_zones_var = {}
flood_edges_var = {}
flood_graph_var = {}

for name in layer_names:
    result = clip_flood_zone(edges.crs, name, neighbors_2_area)
    flood_zones_var[name] = result
    result_1, result_2 = tag_flooded_roads(edges, nodes, result, name)
    flood_edges_var[name] = result_1
    flood_graph_var[name] = result_2
del result, result_1, result_2

layer_names = ["DANA_31_10_2024","DANA_03_11_2024","DANA_05_11_2024","DANA_06_11_2024","DANA_08_11_2024"]
depth_zones = {}

for name in layer_names: 
    depth = flood_depth_zones(name)
    depth_zones[name]=depth

from sys import getsizeof
import types

var_info = []
for name, val in list(globals().items()):
    if (
        not callable(val)  # Skip functions/methods
        and not name.startswith("_")  # Skip private/internal vars
        and not isinstance(val, types.ModuleType)  # Skip imported packages
    ):
        try:
            size = getsizeof(val)
        except Exception:
            size = -1  # Fallback if getsizeof fails
        var_type = type(val).__name__
        var_info.append((name, var_type, size))

# Sort by size (descending)
var_info.sort(key=lambda x: x[2], reverse=True)

# Print table
print(f"{'Variable':<20}{'Type':<20}{'Size (bytes)':<15}")
print("-" * 55)
for name, var_type, size in var_info:
    print(f"{name:<20}{var_type:<20}{size:<15}")

# NAVEGABILITY ANALYSIS
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

def compute_risk_factor_2(T_P_dict, T_NP_dict): #when not computing paths
    keys = set(T_P_dict.keys()) & set(T_NP_dict.keys())
    R = 0
    for k in keys:
        T_P_time = T_P_dict[k]
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

filename = shortest_path_files["Normal Conditions"]

special_nodes = [n for n, attr in G_2nd.nodes(data=True) if attr.get('municipality')]
node_to_muni = {n: attr.get('municipality', '') for n, attr in G_2nd.nodes(data=True)}

T_NP_dictionary = load_or_compute_shortest_paths(filename, G_2nd, special_nodes, node_to_muni,True,True)

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

used_edges = set()

for path, _ in T_NP_dictionary.values():
    for u, v in zip(path, path[1:]):
        used_edges.add((u, v))

%%prun
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

    del (
    G_ig,
    node_to_index,
    index_to_node,
    edge_weights,
    special_indices,
    edge_risks_path,
    base_risk,
    edge_risks,
    i,
    total,
    u,
    v,
    u_idx,
    v_idx,
    eid,
    T_P_dictionary,
    new_risk,
    delta_risk,
    percent_complete,
    edge_risks_json,
    f,
)
    
    %%prun

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

    del (
        G_ig_dana,
        node_to_index,
        index_to_node,
        special_indices,
        T_P_dictionary,
        base_risk,
        candidate_edges,
        edge_risks,
        total,
        i,
        u, v, _, row, weight,
        u_idx, v_idx,
        eid,
        new_risk,
        delta_risk,
        edge_risks_json,
        percent_complete
    )


#SERVICES ACCES

def compute_nearest_tipoelto(G_nx, tipo_elto, output_filename):

    # Convert to igraph
    logging.info("Converting NetworkX graph to iGraph...")
    G_ig, node_to_index, index_to_node, edge_weights = convert_nx_to_igraph(G_nx, weight_attr='travel_time')

    # Identify municipalities and TIPO_ELTO nodes
    muni_nodes = {n: data.get("municipality", "") for n, data in G_nx.nodes(data=True) if data.get("municipality")}
    tipo_nodes = [n for n, data in G_nx.nodes(data=True) if str(data.get("TIPO_ELTO", "")).lower() == tipo_elto.lower()]

    if not tipo_nodes:
        logging.warning(f"No nodes found with TIPO_ELTO='{tipo_elto}'.")
        return None

    logging.info(f"Found {len(muni_nodes)} municipalities and {len(tipo_nodes)} nodes of type '{tipo_elto}'.")

    # Map to igraph indices
    tipo_indices = [node_to_index[n] for n in tipo_nodes]
    result = {}

    # For each municipality, compute shortest path to the nearest tipo_elto node
    for muni_node, muni_name in muni_nodes.items():
        src_idx = node_to_index[muni_node]
        distances = G_ig.distances(src_idx, tipo_indices, weights="weight")[0]

        # Get nearest target
        min_dist = math.inf
        min_idx = None
        for i, d in enumerate(distances):
            if d < min_dist:
                min_dist = d
                min_idx = tipo_indices[i]

        if math.isinf(min_dist):
            logging.warning(f"No reachable {tipo_elto} node for municipality '{muni_name}'.")
            result[muni_name] = {"closest_node": 0, "travel_time": 0}
        else:
            closest_node = index_to_node[min_idx]
            result[muni_name] = {"closest_node": closest_node, "travel_time": float(min_dist)}

    # Save results
    with open(output_filename, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved nearest '{tipo_elto}' nodes for each municipality to {output_filename}")
    return result


output_file = os.path.join(output_dir, "nearest_sanidad.json")
nearest_sanidad = compute_nearest_tipoelto(G_2nd, tipo_elto="Sanidad", output_filename=output_file)

for name, graph in flood_graph_var.items():
    output_file = os.path.join(output_dir, "nearest_sanidad"+str(name)+".json")
    nearest_sanidad = compute_nearest_tipoelto(G_2nd, tipo_elto="Sanidad", output_filename=output_file)

output_file = os.path.join(output_dir, "nearest_seguridad.json")
nearest_sanidad = compute_nearest_tipoelto(G_2nd, tipo_elto="Seguridad", output_filename=output_file)
for name, graph in flood_graph_var.items():
    output_file = os.path.join(output_dir, "nearest_sanidad"+str(name)+".json")
    nearest_sanidad = compute_nearest_tipoelto(G_2nd, tipo_elto="Sanidad", output_filename=output_file)

# Merge all polygons into one MultiPolygon
area_polygon = unary_union(neighbors_2_area.geometry)

# Tags to fetch
tags = {
    "amenity": True,
    "healthcare": True,
    "public_transport": True
}

# Fetch all features in a single query
services = ox.features_from_polygon(area_polygon, tags=tags)

services = services[services.geometry.type == "Point"].copy()

# Keep only points and create a single tidy GeoDataFrame
services = gpd.GeoDataFrame(
    pd.concat(
        [
            services[services[key].notna()]
            .rename(columns={key: "subcategory"})
            .assign(category=key)[["name", "category", "subcategory", "geometry"]]
            for key in tags
        ],
        ignore_index=True
    ),
    crs=neighbors_2_area.crs
)

services.head()

#PLOTS
sorted_data = dict(sorted(R.items(), key=lambda item: item[1]))
names=['High Prob.', 'Med. Prob.', 'Low Prob.', '31/10/2024', '03/11/2024', '05/11/2024', '06/11/2024', '08/11/2024']
values = list(R.values())

plt.figure(figsize=(8, 5))
plt.bar(names, values, color="Blue", alpha=0.5)

plt.ylim(0, 1)
plt.ylabel('$R_{G}$')
#plt.title('Global Risk Index')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('results/Risk_Factor.png')
plt.show()

color_palette = {
    # yr values — warm and spaced
    "10 yr": "#FFD700",    # Gold
    "100 yr": "#FF7F00",   # Dark Orange
    "500 yr": "#B22222",   # Firebrick 

    # DANA values — distinct, avoiding orange/red hues
    "DANA_31_10_2024": "#8A2BE2",  # Blue-Violet
    "DANA_03_11_2024": "#FF1493",  # Deep Pink
    "DANA_05_11_2024": "#00CED1",  # Dark Turquoise
    "DANA_06_11_2024": "#32CD32",  # Lime Green
    "DANA_08_11_2024": "#1E90FF",  # Dodger Blue

    # Normal condition
    "Normal Conditions": "#808080"  # Grey
}

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator

# --- Prepare data ---
plot_data = []
zero_counts = {}
all_counts = {}
layer_names = ["10 yr", "100 yr", "500 yr"]
scenario_names =['High Prob.', 'Med. Prob.', 'Low Prob.', 'Unperturbed']

# T_P_lists scenarios
for name, results_dict in T_P_dictionaries.items():
    if name in layer_names:
        travel_times = [v[1] for v in results_dict.values()]
        non_zero_times = [t / 60 for t in travel_times if t > 0]
        all_times = [t / 60 for t in travel_times]
        zero_counts[name] = travel_times.count(0)
        all_counts[name] = len(travel_times)
        plot_data.append((name, non_zero_times))

# Add normal scenario
normal_travel_times = [v[1] for v in T_NP_dictionary.values()]
normal_non_zero = [t / 60 for t in normal_travel_times if t > 0]
normal_total = len(normal_travel_times)
plot_data.append(("Normal Conditions", normal_non_zero))
all_counts["Normal Conditions"] = normal_total
zero_counts["Normal Conditions"] = normal_travel_times.count(0)

# --- KDE Plot ---
fig, (ax_kde, ax_bar) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2.5, 1]})

x_vals = np.linspace(0, max([max(times) if times else 0 for _, times in plot_data]), 500)

i=0
for name, times in plot_data:
    if len(times) > 1:
        kde = gaussian_kde(times)
        y = kde(x_vals)
        ratio = len(times) / all_counts[name] if all_counts[name] > 0 else 0
        y_rescaled = y * ratio
        color = color_palette.get(name, 'gray')  # fallback to gray if not defined
        ax_kde.plot(x_vals, y_rescaled, label=scenario_names[i], color=color)
    i+=1

ax_kde.set_xlim(0, max([max(times) if times else 0 for _, times in plot_data]))
ax_kde.set_ylim(bottom=0)

ax_kde.set_title("PDF of Travel Times")
ax_kde.set_xlabel("Travel Time (minutes)")
ax_kde.grid(True)
ax_kde.legend(loc='upper right')

# --- Reachability Bar Chart ---
scenarios = layer_names
scenario_names=['High Prob.', 'Med. Prob.', 'Low Prob.']
reachable = [(all_counts[n] - zero_counts[n]) / all_counts[n] if all_counts[n] > 0 else 0 for n in scenarios]
unreachable = [1 - r for r in reachable]

bar_positions = np.arange(len(scenarios))
ax_bar.barh(bar_positions, unreachable, color='salmon', label='Unreachable')
ax_bar.barh(bar_positions, reachable, left=unreachable, color='mediumseagreen', label='Reachable')

ax_bar.set_yticks(bar_positions)
ax_bar.set_yticklabels(scenario_names)
ax_bar.set_xlim(0, 1)
ax_bar.set_xlabel("Fraction of cut routes")
#ax_bar.legend(loc='lower right')
ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

ax_bar.invert_yaxis()

plt.tight_layout()

plt.savefig('results/Travel_times_yr.png')

plt.show()

# --- Prepare data ---
plot_data = []
zero_counts = {}
all_counts = {}
layer_names = ["DANA_31_10_2024","DANA_03_11_2024","DANA_05_11_2024","DANA_08_11_2024"]
scenario_names=['31/10/2024', '03/11/2024', '05/11/2024','08/11/2024','Unperturbed']

for name, results_dict in T_P_dictionaries.items():
    if name in layer_names:
        travel_times = [v[1] for v in results_dict.values()]
        non_zero_times = [t / 60 for t in travel_times if t > 0]
        all_times = [t / 60 for t in travel_times]
        zero_counts[name] = travel_times.count(0)
        all_counts[name] = len(travel_times)
        plot_data.append((name, non_zero_times))

# Add normal scenario
normal_travel_times = [v[1] for v in T_NP_dictionary.values()]
normal_non_zero = [t / 60 for t in normal_travel_times if t > 0]
normal_total = len(normal_travel_times)
plot_data.append(("Normal Conditions", normal_non_zero))
all_counts["Normal Conditions"] = normal_total
zero_counts["Normal Conditions"] = normal_travel_times.count(0)

# --- KDE Plot ---
fig, (ax_kde, ax_bar) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2.5, 1]})

x_vals = np.linspace(0, max([max(times) if times else 0 for _, times in plot_data]), 500)

i = 0
for name, times in plot_data:
    if len(times) > 1:
        kde = gaussian_kde(times)
        y = kde(x_vals)
        ratio = len(times) / all_counts[name] if all_counts[name] > 0 else 0
        y_rescaled = y * ratio
        color = color_palette.get(name, 'gray')  # fallback to gray if not defined
        ax_kde.plot(x_vals, y_rescaled, label=scenario_names[i], color=color)
    i += 1

ax_kde.set_xlim(0, max([max(times) if times else 0 for _, times in plot_data]))
ax_kde.set_ylim(bottom=0)

ax_kde.set_xlabel("Travel Time (minutes)", fontsize=24)
ax_kde.grid(True)
ax_kde.legend(loc='upper right', fontsize=20)
ax_kde.tick_params(axis='both', labelsize=20)

ax_kde.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
ax_kde.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))

# --- Reachability Bar Chart ---
scenarios = layer_names
scenario_names=['31/10/2024', '03/11/2024', '05/11/2024', '08/11/2024']
reachable = [(all_counts[n] - zero_counts[n]) / all_counts[n] if all_counts[n] > 0 else 0 for n in scenarios]
unreachable = [1 - r for r in reachable]

bar_positions = np.arange(len(scenarios))
ax_bar.barh(bar_positions, unreachable, color='salmon', label='Unreachable')
ax_bar.barh(bar_positions, reachable, left=unreachable, color='mediumseagreen', label='Reachable')

ax_bar.set_yticks(bar_positions)
ax_bar.set_yticklabels(scenario_names, fontsize=20)
ax_bar.set_xlim(0, 1)
ax_bar.set_xlabel("Fraction of cut routes", fontsize=24)
ax_bar.tick_params(axis='x', labelsize=20)
ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

ax_bar.invert_yaxis()

plt.tight_layout()

plt.savefig('results/Travel_times_DANA.pdf', format='pdf', dpi=300)
plt.savefig('results/Travel_times_DANA.png', dpi=300)
plt.show()

# MAPS
def add_flood_zone_layer(name, m):
    gdf=flood_zones_var[name]
    color=color_palette[name]
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf_serializable = make_geojson_safe(gdf)

    style_function = lambda x: {
        'fillColor': color,
        'color': color,
        'weight': 1,
        'fillOpacity': 0.4
    }

    geojson = folium.GeoJson(
        data=gdf_serializable,
        name=f"Flood {name}",
        style_function=style_function,
        show=False
    )
    geojson.add_to(m)

    def add_roads_layer(name, m, flood):
    if flood:
        roads = flood_edges_var[name].copy()
        roads = roads[roads["in_flood_zone"] == flood]
    else:
        roads = edges.copy()

    roads = roads.to_crs(epsg=4326)
    roads = make_geojson_safe(roads)

    style_function = lambda x: {
        'color': color_palette[name],
        'weight': 2,
        'opacity': 0.6
    }

    if flood == True:
        geojson = folium.GeoJson(
            roads,
            name=f"Flooded Roads {name}",
            style_function=style_function,
            show=False
        )
    else:
            geojson = folium.GeoJson(
            roads,
            name=f"All roads",
            style_function=style_function,
            show=False
        )
    
    geojson.add_to(m)

    def add_roads_layer_risk(name, m, flood):
    if flood:
        roads = flood_edges_var[name].copy()
        roads = roads[roads["in_flood_zone"] == flood]
    else:
        roads = edges.copy()

    roads = roads.to_crs(epsg=4326)
    roads = make_geojson_safe(roads)

    style_function = lambda x: {
        'color': color_palette[name],
        'weight': 2,
        'opacity': 0.6
    }

    if flood == True:
        geojson = folium.GeoJson(
            roads,
            name=f"Flooded Roads {name}",
            style_function=style_function,
            show=False
        )
    else:
            geojson = folium.GeoJson(
            roads,
            name=f"All roads",
            style_function=style_function,
            show=False
        )
    
    geojson.add_to(m)

    # Set initial position
projected = neighbors_2_area.to_crs(epsg=25830)
centroid_projected = projected.geometry.centroid.iloc[0]
centroid_latlon = gpd.GeoSeries([centroid_projected], crs=25830).to_crs(epsg=4326).geometry.iloc[0]
map_center = [centroid_latlon.y, centroid_latlon.x]
bounds_wgs84 = neighbors_2_area.to_crs(epsg=4326).total_bounds
map_bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
polygon = unary_union(neighbors_2_area.geometry)

m_1 = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron", max_bounds=True)
m_1.fit_bounds(map_bounds)

if os.path.exists(graph_path):
    logging.info("Loading saved road network graph...")
    G = ox.load_graphml(graph_path)
else:
    logging.info("Downloading road network...")
    G = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
    ox.save_graphml(G, filepath=graph_path)
    logging.info("Graph saved.")

nodes, edges = ox.graph_to_gdfs(G)
logging.info("Converted graph to GeoDataFrames.")

# Add flood zones
add_flood_zone_layer("10 yr", m_1)
add_flood_zone_layer("100 yr", m_1)
add_flood_zone_layer("500 yr", m_1)
add_flood_zone_layer("DANA_31_10_2024", m_1)
    
# Add flooded roads (optional)
add_roads_layer("10 yr", m_1, True)
add_roads_layer("100 yr", m_1, True)
add_roads_layer("500 yr", m_1, True)
add_roads_layer("DANA_31_10_2024", m_1, True)
add_roads_layer("Normal Conditions", m_1, False)

folium.LayerControl(collapsed=False).add_to(m_1)
m_1.save("results/Risk_max_DANA.html")
del m_1

import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar

for k, gdf in flood_zones_var.items():
    flood_zones_var[k] = gdf.to_crs(epsg=3857)

# Compute common bounding box across all flood zones
all_bounds = gpd.GeoSeries([gdf.unary_union for gdf in flood_zones_var.values()], crs=3857)
xmin, ymin, xmax, ymax = all_bounds.total_bounds
pad = 500  # meters
dpi=300
extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]

for name, gdf in flood_zones_var.items():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot flood polygons
    gdf.plot(ax=ax, color=color_palette[name], alpha=0.4, edgecolor="k", linewidth=0.1)
    
    # Fix extent
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=12, attribution=False)
    
    # Scale bar (top-right, slightly lower than top)
    scalebar_len = 20000  # 20 km
    x0, y0 = extent[1] - scalebar_len - 15000, extent[3]- 5000
    ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
    ax.text(x0 + scalebar_len/2, y0 - 2000, '20 km', ha='center', va='top', fontsize=10)
    
    # North arrow (top-right corner, slightly above scale bar)
    ax.annotate('', xy=(0.95, 0.92), xytext=(0.95, 0.85), xycoords='axes fraction', 
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=8))
    ax.text(0.95, 0.92, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Style
    ax.axis("off")
    
    # Save PNG
    plt.savefig(f"results/map_{name.replace(' ','_')}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

m_2 = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron", max_bounds=True)
m_2.fit_bounds(map_bounds)

depth=depth_zones["DANA_31_10_2024"]

min_depth = depth["depth_val"].min()
max_depth = depth["depth_val"].max()
depth_colormap = linear.YlGnBu_09.scale(min_depth, max_depth)
depth_colormap.caption = 'Flood Depth (m)'

folium.GeoJson(
    depth,
    name="DANA flood depth",
    style_function=lambda feature: {
        'fillColor': depth_colormap(feature['properties']['depth_val']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(fields=["depth_val"], aliases=["Depth (m):"])
).add_to(m_2)

depth_colormap.add_to(m_2)

folium.LayerControl(collapsed=False).add_to(m_2)
m_2.save("results/Max_flood_depth.html")

m_3 = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron", max_bounds=True)
m_3.fit_bounds(map_bounds)

# Add flood zones
add_flood_zone_layer("DANA_31_10_2024", m_3)
add_flood_zone_layer("DANA_03_11_2024", m_3)
add_flood_zone_layer("DANA_05_11_2024", m_3)
add_flood_zone_layer("DANA_06_11_2024", m_3)
add_flood_zone_layer("DANA_08_11_2024", m_3)
    
# Add flooded roads (optional)
add_roads_layer("DANA_31_10_2024", m_3, True)
add_roads_layer("DANA_03_11_2024", m_3, True)
add_roads_layer("DANA_05_11_2024", m_3, True)
add_roads_layer("DANA_06_11_2024", m_3, True)
add_roads_layer("DANA_08_11_2024", m_3, True)
add_roads_layer("Normal Conditions", m_3, False)

folium.LayerControl(collapsed=False).add_to(m_3)
m_3.save("results/DANA_evolution.html")

def build_delta_dict(delta_list):
    return {
        tuple(item['edge']): item['delta_risk']
        for item in delta_list
    }

def signed_log_transform(x):
    if x == 0:
        return 0
    return 1/(np.sign(x) * np.log10(abs(x)))

def inverted_colormap(val):
    """
    Maps log-transformed delta to color:
    - stronger values (larger abs(val)) → stronger color (closer to red/blue)
    - near-zero values → white
    """
    if val == 0:
        return "#ffffff"
    
    norm_val = signed_log_transform(val) / max_abs_log
    abs_norm = abs(norm_val)

    # invert strength: low abs → white, high abs → strong
    strength = abs_norm
    if norm_val > 0:
        # red side
        return bcm.linear.Reds_09.scale(0, 1)(strength)
    else:
        # blue side
        return bcm.linear.Blues_09.scale(0, 1)(strength)

def add_edges_to_map(fmap, edges, delta_dict, label, filter_flood=False):
    fg = folium.FeatureGroup(name=label)

    for idx, row in edges.iterrows():
        edge_key = (row['u'], row['v']) if 'u' in row and 'v' in row else (idx[0], idx[1])

        if filter_flood and not row.get("in_flood_zone", False):
            continue

        value = delta_dict.get(edge_key)
        if value is None:
            continue

        # Determine color
        if value == 0:
            color = "#bbbbbb"
            opacity = 0.4
            weight = 1
        else:
            color = inverted_colormap(value)
            opacity = 0.8
            weight = 2

        coords = [(lat, lon) for lon, lat in row.geometry.coords]

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=f"{edge_key} | Δ Risk: {value:.1e}"
        ).add_to(fg)

    fg.add_to(fmap)

with open("processed_files/edge_risks_NP.json", "r") as f:
    delta_risks1 = json.load(f)
with open("processed_files/edge_risks_DANA.json", "r") as f:
    delta_risks2 = json.load(f)

delta_dict1 = build_delta_dict(delta_risks1)
delta_dict2 = build_delta_dict(delta_risks2)

all_values = list(delta_dict1.values()) + list(delta_dict2.values())
transformed_values = [signed_log_transform(v) for v in all_values if v is not None]
max_abs_log = max(abs(val) for val in transformed_values)

legend_colormap = bcm.LinearColormap(
    colors=["red", "white", "blue"],
    vmin=-max_abs_log,
    vmax=+max_abs_log
)
legend_colormap.caption = "Δ Risk (Signed Log Scale, Stronger = Darker)"

# === Build map ===
m = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron", max_bounds=True)
m.fit_bounds(map_bounds)

add_edges_to_map(m, edges, delta_dict1, label="All Roads")
add_edges_to_map(m, flood_edges_var["DANA_31_10_2024"], delta_dict2, label="DANA_31_10_2024", filter_flood=True)

m.add_child(legend_colormap)
folium.LayerControl().add_to(m)

m.save("results/delta_risk_map.html")
del m

import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

edges_web = edges.to_crs(3857)
flood_edges_web = flood_edges_var["DANA_31_10_2024"].to_crs(3857)

def plot_roads_map(edges, delta_dict, title, filename, filter_flood=False):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot roads
    for idx, row in edges.iterrows():
        edge_key = (row['u'], row['v']) if 'u' in row and 'v' in row else (idx[0], idx[1])

        if filter_flood and not row.get("in_flood_zone", False):
            continue

        value = delta_dict.get(edge_key)
        if value is None:
            continue

        #if value == 0:
            #color = "#bbbbbb"
            #linewidth = 0.5
        else:
            color = inverted_colormap(value)
            linewidth = 1.2

        xs, ys = row.geometry.xy
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.9)

    # Fix extent with padding
    xmin, ymin, xmax, ymax = edges.total_bounds
    pad = 5000
    extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels,
                    zoom=10, attribution=False)

    # ----- Scale bar + North arrow -----
    scalebar_len = 20000  # 20 km
    x0, y0 = extent[1] - scalebar_len - 15000, extent[3] - 20000
    ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
    ax.text(x0 + scalebar_len/2, y0 - 2000, '20 km',
            ha='center', va='top', fontsize=9)
    ax.annotate('', xy=(x0 + scalebar_len/2, y0 + 12000),
                xytext=(x0 + scalebar_len/2, y0 + 4000),
                arrowprops=dict(facecolor='black', edgecolor='black',
                                width=2, headwidth=8))
    ax.text(x0 + scalebar_len/2, y0 + 12000,
            'N', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

    # ----- Small horizontal colorbar -----
    norm = Normalize(vmin=-max_abs_log, vmax=max_abs_log)
    cmap = plt.cm.bwr
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cax = inset_axes(ax, width="20%", height="2%", loc="upper right",
                     bbox_to_anchor=(-0.05, -0.15, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)

    # Style
    ax.set_axis_off()
    ax.margins(0)
    fig.subplots_adjust(0, 0, 1, 1)

    # Save with no white borders
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# === Generate both styled maps ===
plot_roads_map(edges_web, delta_dict1, "All Roads", "results/all_roads_map.png")
plot_roads_map(flood_edges_web, delta_dict2, "DANA 31-10-2024", "results/dana_map.png", filter_flood=True)


from shapely.geometry import mapping
colormap = linear.YlOrRd_09.scale(0, 1)
colormap.caption = 'Risk Index'

polygons=affected_area

# Create base map
map_center = [centroid_latlon.y, centroid_latlon.x]
m = folium.Map(location=map_center, zoom_start=7, tiles="cartodbpositron")

# Add municipality polygons with color-coded fill
for _, row in polygons.iterrows():
    name = row['name']
    #if name == "Castelló de la Ribera":
        #name = "Castelló de la ribera"
    geometry = row['geometry']
    value = compute_municipal_risk_factor(T_P_dictionaries["DANA_31_10_2024"], T_NP_dictionary, name)
    color = colormap(value)

    geo_json = mapping(geometry)
    folium.GeoJson(
        geo_json,
        tooltip=folium.Tooltip(f"{name}: {value:.2f}"),
        style_function=lambda feature, color=color: {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    ).add_to(m)

    # Add label at centroid
    centroid = geometry.centroid
    folium.Marker(
        location=[centroid.y, centroid.x],
        icon=folium.DivIcon(html=f"""<div style="font-size: 10pt; color: black;">{name}</div>""")
    ).add_to(m)

# Add color map legend
colormap.add_to(m)

# Display map
m.save("results/municipality_risk_map.html")
del m

import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Reproject polygons to Web Mercator for basemap compatibility
polygons = affected_area.to_crs(epsg=3857)
name_corrections = {
    "Castelló de la Ribera":"Castelló de la ribera"
}

# Compute risk values with name correction
polygons["risk"] = polygons["name"].apply(
    lambda name: compute_municipal_risk_factor(
        T_P_dictionaries["DANA_31_10_2024"], 
        T_NP_dictionary, 
        name_corrections.get(name, name)  # use corrected name if exists
    )
)

aux = polygons[['name','risk']].values
print(aux)

# Colormap scaling
norm = Normalize(vmin=polygons["risk"].min(), vmax=polygons["risk"].max())
cmap = cm.get_cmap("YlOrRd")

# Figure setup
fig, ax = plt.subplots(figsize=(10, 10))

# Plot polygons
polygons.plot(
    column="risk",
    cmap=cmap,
    linewidth=0.5,
    edgecolor="black",
    ax=ax,
    legend=False
)

# Fix extent with padding
xmin, ymin, xmax, ymax = polygons.total_bounds
pad = 5000
extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels,
                zoom=10, attribution=False)

# Add labels at centroids, store for adjustment
texts = []
for idx, row in polygons.iterrows():
    centroid = row.geometry.centroid
    txt = ax.text(
        centroid.x, centroid.y, row["name"],
        fontsize=8, ha="center", color="black"
    )
    # Thinner white halo
    txt.set_path_effects([
        path_effects.Stroke(linewidth=0.8, foreground="white"),
        path_effects.Normal()
    ])
    texts.append(txt)

# Adjust labels to avoid overlaps (with leader lines)
adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    only_move={'points':'xy', 'text':'xy'},
    expand_text=(5, 5),
    expand_points=(5, 5),
    force_text=0.5,
    force_points=0.5,
    precision=0.0001,
    lim=5000,
    autoalign='xy'
)

# ----- Scalebar + North arrow (top-right corner) -----
scalebar_len = 20000  # 20 km
x0, y0 = extent[1] - scalebar_len - 15000, extent[3] - 20000
ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
ax.text(x0 + scalebar_len/2, y0 - 2000, '20 km',
        ha='center', va='top', fontsize=9)

# North arrow above scale bar
ax.annotate('', xy=(x0 + scalebar_len/2, y0 + 12000),
            xytext=(x0 + scalebar_len/2, y0 + 4000),
            arrowprops=dict(facecolor='black', edgecolor='black',
                            width=2, headwidth=8))
ax.text(x0 + scalebar_len/2, y0 + 12000,
        'N', ha='center', va='bottom',
        fontsize=11, fontweight='bold')

# ----- Small colorbar just below them -----
cax = inset_axes(ax, width="20%", height="2%", loc="upper right",
                 bbox_to_anchor=(-0.05, -0.15, 1, 1),
                 bbox_transform=ax.transAxes, borderpad=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
cbar.ax.tick_params(labelsize=8)

# Style
ax.set_axis_off()
ax.margins(0)
fig.subplots_adjust(0, 0, 1, 1)

# Save with no white borders
plt.savefig("results/municipality_risk_map.png",
            dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import contextily as ctx
import geopandas as gpd
import os

def plot_selected_tipoelto_points_map(points_gdf, tipo_list, attr_col, title, output_dir):
    """
    Plot selected TIPO_ELTO categories (points) on a basemap with consistent style.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        GeoDataFrame of nodes or points with a geometry column (Point)
        and an attribute column (e.g., 'TIPO_ELTO').
    tipo_list : list of str
        List of TIPO_ELTO values to include (e.g. ['sanidad', 'educacion']).
    attr_col : str
        Column name representing the type of the point (e.g., 'TIPO_ELTO').
    title : str
        Title of the plot.
    output_dir : str
        Folder where the map should be saved (e.g., 'results/').
    """

    # --- 1️⃣ Filter valid geometries and selected TIPO_ELTO values ---
    import unicodedata

    # --- Normalize both the dataset and the filter list ---
    def normalize_str(s):
        if not isinstance(s, str):
            return ""
        s = s.lower()
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
        return s.strip()

    points_gdf = points_gdf[points_gdf.geometry.notnull() & points_gdf[attr_col].notnull()]

    # Normalize attribute values in the dataframe
    points_gdf["__normalized_tipo__"] = points_gdf[attr_col].apply(normalize_str)

    # Normalize your list of desired types
    normalized_tipo_list = [normalize_str(x) for x in tipo_list]

    # Filter case- and accent-insensitive
    points_gdf = points_gdf[points_gdf["__normalized_tipo__"].isin(normalized_tipo_list)]


    if points_gdf.empty:
        print(f"⚠️ No points found with {attr_col} in {tipo_list}")
        return

    # --- 2️⃣ Ensure CRS is Web Mercator ---
    if points_gdf.crs is None or points_gdf.crs.to_epsg() != 3857:
        points_gdf = points_gdf.to_crs(epsg=3857)

    # --- 3️⃣ Prepare figure ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # --- 4️⃣ Prepare color map (one color per TIPO_ELTO) ---
    tipos = sorted(points_gdf[attr_col].unique())
    cmap = get_cmap("tab10", len(tipos))
    colors = {tipo: to_hex(cmap(i)) for i, tipo in enumerate(tipos)}

    # --- 5️⃣ Plot each TIPO_ELTO separately ---
    for tipo, color in colors.items():
        subset = points_gdf[points_gdf[attr_col] == tipo]
        subset.plot(
            ax=ax,
            color=color,
            markersize=30,
            alpha=0.9,
            label=tipo,
            zorder=3,
            marker='o',
            edgecolor='k',
            linewidth=0.3
        )

    # --- 6️⃣ Compute extent and padding ---
    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    pad = (xmax - xmin) * 0.05
    extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # --- 7️⃣ Adaptive zoom ---
    width_m = xmax - xmin
    if width_m > 10_000_000:
        zoom = 5
    elif width_m > 5_000_000:
        zoom = 6
    elif width_m > 2_000_000:
        zoom = 7
    elif width_m > 1_000_000:
        zoom = 8
    else:
        zoom = 10

    # --- 8️⃣ Add basemap ---
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels,
                    zoom=zoom, attribution=False)

    # --- 9️⃣ Scale bar + North arrow ---
    scalebar_len = 20000  # 20 km
    x0, y0 = extent[0] + 20000, extent[2] + 20000
    ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
    ax.text(x0 + scalebar_len / 2, y0 - 2000, '20 km', ha='center', va='top', fontsize=9)
    ax.annotate('', xy=(x0 + scalebar_len / 2, y0 + 12000),
                xytext=(x0 + scalebar_len / 2, y0 + 4000),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=8))
    ax.text(x0 + scalebar_len / 2, y0 + 12000, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- 🔟 Legend & Save ---
    ax.legend(title=attr_col, loc="upper right", frameon=True, fontsize=8)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")

    plt.title(title, fontsize=12)
    plt.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"✅ Map saved to {filepath}")


# Load your graph nodes
nodes, edges = ox.graph_to_gdfs(G_2nd)

selected_types = ["sanidad", "educacion", "seguridad", "transporte","residencial especial", "servicios basicos"]

# Plot
plot_selected_tipoelto_points_map(
    nodes,
    tipo_list=selected_types,
    attr_col="TIPO_ELTO",
    title="Selected Points of Interest",
    output_dir="results"
)

