import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon

# ============================================================
# Configuration
# ============================================================

# Use script directory as origin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "source_files")
output_dir = os.path.join(BASE_DIR, "processed_files")
os.makedirs(output_dir, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

# ============================================================
# Utility Functions
# ============================================================

def parse_depth_range(val):
    if pd.isna(val):
        return None
    val = val.strip()
    if val.startswith('Below'):
        return float(val[5:].strip()) / 2
    if val.startswith('>'):
        return float(val[1:].strip())
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


def clip_flood_zone(return_crs, name, clip_geom):
    """Clip flood polygons to study area and save to GeoPackage."""
    output_path = zone_output_files[name]
    input_path = zone_input_files[name]
    
    if os.path.exists(output_path):
        print(f"Loading {name} from {output_path}")
        clipped = gpd.read_file(output_path, layer=name).to_crs(return_crs)
    else:
        print(f"Clipping and saving {name} to {output_path}")
        flood = gpd.read_file(input_path).to_crs(return_crs)
        clipped = gpd.clip(flood, clip_geom)
        clipped.to_file(output_path, layer=name, driver="GPKG")
    return clipped


def flood_depth_zones(name):
    """Parse flood depth shapefiles and save standardized depth GeoPackage."""
    layer = "depth_val"
    input_path = depth_input_files[name]
    output_path = depth_output_files[name]

    if os.path.exists(output_path):
        print(f"Loading {layer} from {output_path}")
        depth = gpd.read_file(output_path, layer=layer)
    else:
        print(f"Saving {layer} to {output_path}")
        depth = gpd.read_file(input_path)
        depth["depth_val"] = depth["value"].apply(parse_depth_range)
        depth.to_file(output_path, layer=layer, driver="GPKG")
        print(f"Saved processed {layer} in {output_path}")
    return depth


def tag_flooded_roads(edges, nodes, flood_zones, name):
    """
    Tag edges as flooded or safe, and save:
      - full tagged network (.gpkg)
      - flooded (cut) edges (.gpkg + .graphml)
      - safe edges (.gpkg + .graphml)
    """

    tagged_path = os.path.join(output_dir, f"tagged_roads_{name}.gpkg")
    cut_gpkg_path = os.path.join(output_dir, f"cut_roads_{name}.gpkg")
    safe_gpkg_path = os.path.join(output_dir, f"safe_roads_{name}.gpkg")
    cut_graphml_path = os.path.join(output_dir, f"cut_roads_{name}.graphml")
    safe_graphml_path = os.path.join(output_dir, f"safe_roads_{name}.graphml")

    # --- Step 1: Tag flooded edges ---
    if os.path.exists(tagged_path):
        print(f"Loading tagged edges for {name} from {tagged_path}")
        edges = gpd.read_file(tagged_path)

        # ---- Restore MultiIndex ----
        if not isinstance(edges.index, pd.MultiIndex):
            if "u" in edges.columns and "v" in edges.columns:
                if "key" not in edges.columns:
                    edges["key"] = 0
                edges.set_index(["u", "v", "key"], inplace=True)
    else:
        print(f"Tagging flooded edges for {name}")
        bounds = edges.total_bounds
        flood_subset = flood_zones.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        flood_geoms = flood_subset.geometry

        edges = edges.copy()
        edges["in_flood_zone"] = edges.geometry.apply(
            lambda geom: flood_geoms.intersects(geom).any()
        )

        # Save tagged edges
        edges.to_file(tagged_path, layer=name, driver="GPKG")
        print(f"Saved tagged edges to {tagged_path}")

        # Restore MultiIndex
        if not isinstance(edges.index, pd.MultiIndex):
            if "u" in edges.columns and "v" in edges.columns:
                if "key" not in edges.columns:
                    edges["key"] = 0
                edges.set_index(["u", "v", "key"], inplace=True)

    # --- Safe vs Flooded ---
    safe_edges = edges[~edges["in_flood_zone"]].copy()
    cut_edges = edges[edges["in_flood_zone"]].copy()

    # Restore MultiIndex for safe_edges
    if not isinstance(safe_edges.index, pd.MultiIndex):
        if "u" in safe_edges.columns and "v" in safe_edges.columns:
            if "key" not in safe_edges.columns:
                safe_edges["key"] = 0
            safe_edges.set_index(["u", "v", "key"], inplace=True)

    # Restore MultiIndex for cut_edges
    if not isinstance(cut_edges.index, pd.MultiIndex):
        if "u" in cut_edges.columns and "v" in cut_edges.columns:
            if "key" not in cut_edges.columns:
                cut_edges["key"] = 0
            cut_edges.set_index(["u", "v", "key"], inplace=True)

    # --- Save GeoPackages ---
    safe_edges.to_file(safe_gpkg_path, layer=name, driver="GPKG")
    cut_edges.to_file(cut_gpkg_path, layer=name, driver="GPKG")
    print("Saved safe/cut edges to GPKG")

    # ============================================================
    # Preserve municipality and clase ATTRIBUTES
    # ============================================================
    nodes = nodes.copy()
    nodes["municipality"] = [G_2nd.nodes[n].get("municipality", None) for n in nodes.index]
    nodes["clase"] = [G_2nd.nodes[n].get("clase", None) for n in nodes.index]

    # --- Build graphs WITH ATTRIBUTES PRESERVED ---
    G_safe = ox.graph_from_gdfs(nodes, safe_edges)
    G_cut = ox.graph_from_gdfs(nodes, cut_edges)

    # --- Save GraphMLs ---
    ox.save_graphml(G_safe, safe_graphml_path)
    ox.save_graphml(G_cut, cut_graphml_path)
    print("Saved safe/cut roads GraphML WITH municipality + clase attributes")

    return edges, G_safe, G_cut


# ============================================================
# File Paths and Study Area
# ============================================================

FILE_2ND = os.path.join(output_dir, "neighbors_2_area.gpkg")
neighbors_2_area = gpd.read_file(FILE_2ND)
print(f"Loaded study area from {FILE_2ND}")

graph_path = os.path.join(output_dir, "G_2nd.graphml")
G_2nd = ox.load_graphml(graph_path)
print(f"Loaded saved graph from {graph_path}")

nodes, edges = ox.graph_to_gdfs(G_2nd)

# ============================================================
# File Dictionaries
# ============================================================

zone_input_files = {
    "10 yr": os.path.join(input_dir, "laminaspb-q10/Q10_2Ciclo_PB_20241121.shp"),
    "100 yr": os.path.join(input_dir, "laminaspb-q100/Q100_2Ciclo_PB_20241121_ETRS89.shp"),
    "500 yr": os.path.join(input_dir, "laminaspb-q500/Q500_2Ciclo_PB_20241121_ETRS89.shp"),
    "DANA_31_10_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_PRODUCT_v1/EMSR773_AOI01_DEL_PRODUCT_observedEventA_v1.shp"),
    "DANA_03_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT01_v1/EMSR773_AOI01_DEL_MONIT01_observedEventA_v1.shp"),
    "DANA_05_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT02_v1/EMSR773_AOI01_DEL_MONIT02_observedEventA_v1.shp"),
    "DANA_06_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT03_v1/EMSR773_AOI01_DEL_MONIT03_observedEventA_v1.shp"),
    "DANA_08_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT04_v1/EMSR773_AOI01_DEL_MONIT04_observedEventA_v1.shp")
}

zone_output_files = {
    "10 yr": os.path.join(output_dir, "zone_flood_risk_10.gpkg"),
    "100 yr": os.path.join(output_dir, "zone_flood_risk_100.gpkg"),
    "500 yr": os.path.join(output_dir, "zone_flood_risk_500.gpkg"),
    "DANA_31_10_2024": os.path.join(output_dir, "zone_DANA_31_10_2024.gpkg"),
    "DANA_03_11_2024": os.path.join(output_dir, "zone_DANA_03_11_2024.gpkg"),
    "DANA_05_11_2024": os.path.join(output_dir, "zone_DANA_05_11_2024.gpkg"),
    "DANA_06_11_2024": os.path.join(output_dir, "zone_DANA_06_11_2024.gpkg"),
    "DANA_08_11_2024": os.path.join(output_dir, "zone_DANA_08_11_2024.gpkg")
}

depth_input_files = {
    "DANA_31_10_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_PRODUCT_v1/EMSR773_AOI01_DEL_PRODUCT_floodDepthA_v1.shp"),
    "DANA_03_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT01_v1/EMSR773_AOI01_DEL_MONIT01_floodDepthA_v1.shp"),
    "DANA_05_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT02_v1/EMSR773_AOI01_DEL_MONIT02_floodDepthA_v1.shp"),
    "DANA_06_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT03_v1/EMSR773_AOI01_DEL_MONIT03_floodDepthA_v1.shp"),
    "DANA_08_11_2024": os.path.join(input_dir, "EMSR773_AOI01_DEL_MONIT04_v1/EMSR773_AOI01_DEL_MONIT04_floodDepthA_v1.shp")
}

depth_output_files = {
    "DANA_31_10_2024": os.path.join(output_dir, "depth_DANA_31_10_2024.gpkg"),
    "DANA_03_11_2024": os.path.join(output_dir, "depth_DANA_03_11_2024.gpkg"),
    "DANA_05_11_2024": os.path.join(output_dir, "depth_DANA_05_11_2024.gpkg"),
    "DANA_06_11_2024": os.path.join(output_dir, "depth_DANA_06_11_2024.gpkg"),
    "DANA_08_11_2024": os.path.join(output_dir, "depth_DANA_08_11_2024.gpkg")
}

# ============================================================
# Main Execution
# ============================================================

layer_names = [
    "10 yr", "100 yr", "500 yr",
    "DANA_31_10_2024", "DANA_03_11_2024",
    "DANA_05_11_2024", "DANA_06_11_2024", "DANA_08_11_2024"
]

flood_zones_var = {}
flood_edges_var = {}
flood_graph_var = {}

for name in layer_names:
    print(f"\n=== Processing {name} ===")
    flood_zone = clip_flood_zone(edges.crs, name, neighbors_2_area)
    flood_zones_var[name] = flood_zone

    edges_tagged, G_safe, G_cut = tag_flooded_roads(edges, nodes, flood_zone, name)
    flood_edges_var[name] = edges_tagged
    flood_graph_var[name + "_safe"] = G_safe
    flood_graph_var[name + "_cut"] = G_cut

print("\nProcessing complete! All cut/safe roads saved as both .gpkg and .graphml.")
