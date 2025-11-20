import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import os
import sys

# ---------------------------------------------------------
# RESOLVE SCRIPT DIRECTORY
# ---------------------------------------------------------
# Works for PyInstaller executables and normal .py scripts
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Script directory: {SCRIPT_DIR}")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
zones = True
roads = True

output_dir = os.path.join(SCRIPT_DIR, "processed_files")
results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

dpi = 300
pad = 500  # meters

# ---------------------------------------------------------
# COLOR SCHEME
# ---------------------------------------------------------
color_palette = {
    "10 yr": "#FFD700",     # Gold
    "100 yr": "#FF7F00",    # Dark Orange
    "500 yr": "#B22222",    # Firebrick
    "DANA_31_10_2024": "#8A2BE2",  # Blue-Violet
    "DANA_03_11_2024": "#FF1493",  # Deep Pink
    "DANA_05_11_2024": "#00CED1",  # Dark Turquoise
    "DANA_06_11_2024": "#32CD32",  # Lime Green
    "DANA_08_11_2024": "#1E90FF",  # Dodger Blue
    "Normal Conditions": "#808080"  # Grey
}

# ---------------------------------------------------------
# FILE PATHS — now relative to script location
# ---------------------------------------------------------
def path(filename):
    return os.path.join(output_dir, filename)

cut_roads_files = {
    "10 yr": path("cut_roads_10.gpkg"),
    "100 yr": path("cut_roads_100.gpkg"),
    "500 yr": path("cut_roads_500.gpkg"),
    "DANA_31_10_2024": path("cut_roads_DANA_31_10_2024.gpkg"),
    "DANA_03_11_2024": path("cut_roads_DANA_03_11_2024.gpkg"),
    "DANA_05_11_2024": path("cut_roads_DANA_05_11_2024.gpkg"),
    "DANA_06_11_2024": path("cut_roads_DANA_06_11_2024.gpkg"),
    "DANA_08_11_2024": path("cut_roads_DANA_08_11_2024.gpkg")
}

safe_roads_files = {
    "10 yr": path("safe_roads_10.gpkg"),
    "100 yr": path("safe_roads_100.gpkg"),
    "500 yr": path("safe_roads_500.gpkg"),
    "DANA_31_10_2024": path("safe_roads_DANA_31_10_2024.gpkg"),
    "DANA_03_11_2024": path("safe_roads_DANA_03_11_2024.gpkg"),
    "DANA_05_11_2024": path("safe_roads_DANA_05_11_2024.gpkg"),
    "DANA_06_11_2024": path("safe_roads_DANA_06_11_2024.gpkg"),
    "DANA_08_11_2024": path("safe_roads_DANA_08_11_2024.gpkg")
}

zone_output_files = {
    "10 yr": path("zone_flood_risk_10.gpkg"),
    "100 yr": path("zone_flood_risk_100.gpkg"),
    "500 yr": path("zone_flood_risk_500.gpkg"),
    "DANA_31_10_2024": path("zone_DANA_31_10_2024.gpkg"),
    "DANA_03_11_2024": path("zone_DANA_03_11_2024.gpkg"),
    "DANA_05_11_2024": path("zone_DANA_05_11_2024.gpkg"),
    "DANA_06_11_2024": path("zone_DANA_06_11_2024.gpkg"),
    "DANA_08_11_2024": path("zone_DANA_08_11_2024.gpkg")
}

layer_names = [
    "DANA_31_10_2024",
    "DANA_03_11_2024",
    "DANA_05_11_2024",
    "DANA_08_11_2024"
]

# ---------------------------------------------------------
# LOAD FLOOD ZONES
# ---------------------------------------------------------
flood_zones_var = {}

for name in layer_names:
    print(f"Loading flood zone: {name}")
    gdf = gpd.read_file(zone_output_files[name])
    flood_zones_var[name] = gdf.to_crs(epsg=3857)

# Compute shared bounding box across all flood zones
all_bounds = gpd.GeoSeries([gdf.unary_union for gdf in flood_zones_var.values()], crs=3857)
xmin, ymin, xmax, ymax = all_bounds.total_bounds
extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]

# ---------------------------------------------------------
# PLOTTING LOOP
# ---------------------------------------------------------
for name, gdf in flood_zones_var.items():
    print(f"\nPlotting scenario: {name}")
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Flood zones ---
    if zones:
        gdf.plot(ax=ax, color=color_palette[name], alpha=0.4, edgecolor="k", linewidth=0.1)

    # --- Roads ---
    if roads:
        print("Loading roads...")
        if os.path.exists(safe_roads_files[name]) and os.path.exists(cut_roads_files[name]):
            safe_roads_gdf = gpd.read_file(safe_roads_files[name]).to_crs(3857)
            cut_roads_gdf = gpd.read_file(cut_roads_files[name]).to_crs(3857)

            safe_roads_gdf.plot(ax=ax, color="black", linewidth=0.3, alpha=0.4, label="Safe roads")
            cut_roads_gdf.plot(ax=ax, color="red", linewidth=0.3, alpha=0.4, label="Cut roads")
        else:
            print(f"Missing roads for {name}")

    # --- Map styling ---
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=12, attribution=False)

    # Scale bar
    scalebar_len = 20000  # 20 km
    x0, y0 = extent[1] - scalebar_len - 15000, extent[3] - 5000
    ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
    ax.text(x0 + scalebar_len / 2, y0 - 2000, '20 km', ha='center', va='top', fontsize=10)

    # North arrow
    ax.annotate('', xy=(0.95, 0.92), xytext=(0.95, 0.85), xycoords='axes fraction',
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=8))
    ax.text(0.95, 0.92, 'N', transform=ax.transAxes, ha='center', va='bottom',
            fontsize=12, fontweight='bold')

    # Final touches
    ax.axis("off")
    ax.legend(loc="lower left")

    # Save figure
    save_path = os.path.join(results_dir, f"map_{name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved {save_path}")
