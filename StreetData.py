import os
import logging
from itertools import combinations
import json

import numpy as np
import pandas as pd
import geopandas as gpd

import networkx as nx
import osmnx as ox
from shapely.geometry import MultiPolygon
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# FUNCTIONS
# ---------------------------

def download_municipalities(names, region="Valencia, Spain"):
    if not names:
        try:
            region_gdf = ox.geocode_to_gdf(region)
            polygon = region_gdf.geometry.iloc[0]
            gdf = ox.geometries_from_polygon(polygon, tags={"boundary": "administrative"})
            gdf = gdf[gdf["admin_level"] == "8"]
            gdf = gdf[["geometry", "name"]].reset_index(drop=True)
            gdf = gdf[gdf["name"].notna()]
            gdf_clipped = gpd.clip(gdf, polygon)
            gdf_clipped = gdf_clipped[~gdf_clipped.is_empty].reset_index(drop=True)
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
                data["travel_time"] = data["length"] / speed_mps + 5

        ox.save_graphml(G, graph_path)
        logging.info(f"Graph saved to {graph_path}")

    # Assign municipalities to nodes
    nodes, _ = ox.graph_to_gdfs(G)
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
        # Update graph nodes
        for node_id, row in nodes.iterrows():
            G.nodes[node_id]["municipality"] = row["municipality"]
        ox.save_graphml(G, graph_path)
        logging.info(f"Updated graph saved to: {graph_path}")
    else:
        logging.info("Municipality names already exist in graph; skipping reassignment.")

    return G

def assign_single_attribute_to_graph(G, gdf, attr='clase', graph_path=None):
    nodes, _ = ox.graph_to_gdfs(G)
    node_coords = np.array([(geom.y, geom.x) for geom in nodes.geometry])
    kdtree = cKDTree(node_coords)

    if attr not in nodes.columns:
        nodes[attr] = None

    for idx, row in gdf.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        _, nearest_idx = kdtree.query([lat, lon], k=1)
        nearest_node = nodes.index[nearest_idx]
        nodes.at[nearest_node, attr] = row[attr]

    for node_id, row in nodes.iterrows():
        G.nodes[node_id][attr] = row[attr]

    if graph_path:
        ox.save_graphml(G, graph_path)
        logging.info(f"Graph updated with '{attr}' and saved to {graph_path}")

    return G

# ---------------------------
# MAIN WORKFLOW
# ---------------------------

base_path = os.path.dirname(os.path.abspath(__file__))
output_dir = "processed_files"
output_dir= os.path.join(base_path,output_dir)
os.makedirs(output_dir, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

FILE_STUDY = os.path.join(output_dir, "affected_area.gpkg")
FILE_1ST = os.path.join(output_dir, "neighbors_1_area.gpkg")
FILE_2ND = os.path.join(output_dir, "neighbors_2_area.gpkg")
graph_path_2nd = os.path.join(output_dir, "G_2nd.graphml")

json_path = os.path.join(base_path, "source_files", "affected_municipalities_dictionary.json")
with open(json_path, "r", encoding="utf-8") as f:
    regions = json.load(f)
affected_valencia = regions["Valencia"]["affected"]
affected_cuenca = regions["Cuenca"]["affected"]
urban_center = {**regions["Valencia"]["coordinates"], **regions["Cuenca"]["coordinates"]}
del regions

# Load or create study area
if os.path.exists(FILE_STUDY):
    affected_area = gpd.read_file(FILE_STUDY)
else:
    valencia_area = download_municipalities(affected_valencia, "Valencia, Spain")
    cuenca_area = download_municipalities(affected_cuenca, "Cuenca, Spain")
    affected_area = gpd.GeoDataFrame(pd.concat([valencia_area, cuenca_area], ignore_index=True))
    affected_area.to_file(FILE_STUDY, driver="GPKG")

# Load or create neighbors
if os.path.exists(FILE_1ST) and os.path.exists(FILE_2ND):
    neighbors_1_area = gpd.read_file(FILE_1ST)
    neighbors_2_area = gpd.read_file(FILE_2ND)
else:
    regions_list = [
        'Provincia de Valencia, Comunidad Valenciana, Spain', 
        'Provincia de Castellon, Comunidad Valenciana, Spain', 
        'Provincia de Cuenca, Castilla-La Mancha, Spain', 
        'Provincia de Albacete, Castilla-La Mancha, Spain', 
        'Provincia de Teruel, Aragon, Spain'
    ]
    possible_neighbors = gpd.GeoDataFrame()
    for region in regions_list:
        aux = download_municipalities("", region)
        possible_neighbors = pd.concat([possible_neighbors, aux], ignore_index=True)
    possible_neighbors = possible_neighbors[possible_neighbors.geometry.type.isin(["Polygon", "MultiPolygon"]) & possible_neighbors.geometry.is_valid & ~possible_neighbors.is_empty]
    possible_neighbors = possible_neighbors.to_crs(affected_area.crs)

    union_1 = affected_area.geometry.unary_union
    adjacent_1 = possible_neighbors[possible_neighbors.geometry.touches(union_1)]
    neighbors_1_area = gpd.GeoDataFrame(pd.concat([affected_area, adjacent_1], ignore_index=True), crs=affected_area.crs)
    neighbors_1_area.to_file(FILE_1ST, driver="GPKG")

    union_2 = neighbors_1_area.geometry.unary_union
    adjacent_2 = possible_neighbors[possible_neighbors.geometry.touches(union_2)]
    neighbors_2_area = gpd.GeoDataFrame(pd.concat([neighbors_1_area, adjacent_2], ignore_index=True), crs=affected_area.crs)
    neighbors_2_area.to_file(FILE_2ND, driver="GPKG")

# Build graphs
G_2nd = build_graph_from_layer(neighbors_2_area, graph_path_2nd, urban_center)

# Filter by selected clases and assign 'classes' attribute
selected_clases = [
    "Hospital",
    "Otros centros sanitarios",
    "Orden público-seguridad",
    "Emergencias"
]

base_path = os.path.dirname(os.path.abspath(__file__))
gpkg_path = os.path.join(base_path, "source_files", "BTN_POI_Servicios_instalaciones_gpkg", "BTN_POI_Servicios_instalaciones_gpkg.gpkg")
special_interest_points=gpd.read_file(gpkg_path)

filtered_points = special_interest_points[special_interest_points['clase'].isin(selected_clases)].copy()

G_2nd = assign_single_attribute_to_graph(
    G_2nd,
    filtered_points,
    attr='clase',
    graph_path=graph_path_2nd,
)

