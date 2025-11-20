import os
import geopandas as gpd

# Get the directory where the current script is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Build the path to the GeoPackage relative to the script folder
gpkg_path = os.path.join(base_path, "source_files", "BTN_POI_Servicios_instalaciones_gpkg", "BTN_POI_Servicios_instalaciones_gpkg.gpkg")

# Read the GeoPackage
gdf = gpd.read_file(gpkg_path)

# Replace 'type' with the column you want
column_name = 'clase'

# Show first rows
print(gdf.head())
print(gdf.columns)

# Get unique values and counts
counts = gdf[column_name].value_counts()

# Print them
print(counts)

from collections import Counter
import networkx as nx
import os

# Load the GraphML file
gpkg_path = os.path.join(base_path, "processed_files", "G_2nd.graphml")
G = nx.read_graphml(gpkg_path)

# Extract all values of the 'clase' attribute for nodes
clase_values = [data.get('clase') for _, data in G.nodes(data=True)]

# Count occurrences of each class
clase_counts = Counter(clase_values)

# Print the counts
print("Counts for each 'clase':")
for clase, count in clase_counts.items():
    print(f"{clase}: {count}")