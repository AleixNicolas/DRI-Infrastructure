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
from scipy.spatial import cKDTree  # <-- ADD THIS LINE

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

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
