import geopandas as gpd

cut = gpd.read_file("processed_files/cut_roads_DANA_31_10_2024.gpkg")
safe = gpd.read_file("processed_files/safe_roads_DANA_31_10_2024.gpkg")

print("Cut roads:", len(cut))
print("Safe roads:", len(safe))
print("Overlap:", cut.geometry.equals(safe.geometry))
print("Columns:", cut.columns)
print(cut.head())