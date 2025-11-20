import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import contextily as ctx
import osmnx as ox

# -------------------- Helper Functions --------------------
def compute_individual_risk_factor(T_P, T_NP):
    """
    Compute risk factor for a single path.
    Handles missing values (None) safely.
    """

    # No travel possible under perturbed conditions → maximum risk
    if T_P is None:
        return 1

    # No baseline travel time (not connected normally) → no risk
    if T_NP is None:
        return 0

    # Avoid division by zero
    if T_P == 0:
        return 1

    return 1 - (T_NP / T_P)


def compute_municipal_risk_factor(T_P_dict, T_NP_dict, municipality):
    """Compute average risk factor for a municipality."""
    keys = {k for k in T_P_dict.keys() & T_NP_dict.keys() if municipality in k}

    if not keys:
        return 0

    R = 0
    for k in keys:
        _, T_P = T_P_dict[k]
        _, T_NP = T_NP_dict[k]
        R += compute_individual_risk_factor(T_P, T_NP)

    return R / len(keys)


def load_shortest_paths(filename):
    """
    Load shortest paths JSON into dictionary.
    Returns: key → (path_list, time_value)
    Handles {"path": [...], "time": ...} structure.
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    cleaned = {}
    for k, v in data.items():
        path = v.get("path", [])
        time = v.get("time", None)
        cleaned[k] = (path, time)

    return cleaned


# -------------------- Paths & Data --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "processed_files")

FILE_STUDY = os.path.join(output_dir, "affected_area.gpkg")
affected_area = gpd.read_file(FILE_STUDY)
print(f"Loaded study area from {FILE_STUDY}")

# Reproject polygons to Web Mercator
polygons = affected_area.to_crs(epsg=3857)

# Load graph
FILE_GRAPH = os.path.join(output_dir, "G_2nd.graphml")
G_2nd = ox.load_graphml(FILE_GRAPH)

# Load shortest paths (correctly parsed)
T_NP_dictionary = load_shortest_paths(os.path.join(output_dir, "shortest_paths_NP.json"))
T_P_dictionaries = {
    "DANA_31_10_2024": load_shortest_paths(os.path.join(output_dir, "shortest_paths_DANA_31_10_2024.json"))
}

# Name corrections for matching
name_corrections = {
    "Castelló de la Ribera": "Castelló de la ribera"
}

# -------------------- Compute Risk --------------------
polygons["risk"] = polygons["name"].apply(
    lambda name: compute_municipal_risk_factor(
        T_P_dictionaries["DANA_31_10_2024"],
        T_NP_dictionary,
        name_corrections.get(name, name)
    )
)

print(polygons[['name', 'risk']].values)

# -------------------- Plotting --------------------
norm = Normalize(vmin=polygons["risk"].min(), vmax=polygons["risk"].max())
cmap = cm.get_cmap("YlOrRd")

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

# Set extent with padding
xmin, ymin, xmax, ymax = polygons.total_bounds
pad = 5000
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels,
                zoom=10, attribution=False)

# Add labels at centroids
texts = []
for idx, row in polygons.iterrows():
    centroid = row.geometry.centroid
    txt = ax.text(
        centroid.x, centroid.y, row["name"],
        fontsize=8, ha="center", color="black"
    )
    txt.set_path_effects([
        path_effects.Stroke(linewidth=0.8, foreground="white"),
        path_effects.Normal()
    ])
    texts.append(txt)

# Adjust labels
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

# -------------------- Scalebar & North Arrow --------------------
scalebar_len = 20000  # 20 km
x0, y0 = xmax - scalebar_len - 15000, ymax - 20000

# Scale bar
ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
ax.text(x0 + scalebar_len/2, y0 - 2000, '20 km',
        ha='center', va='top', fontsize=9)

# North arrow
ax.annotate('', xy=(x0 + scalebar_len/2, y0 + 12000),
            xytext=(x0 + scalebar_len/2, y0 + 4000),
            arrowprops=dict(facecolor='black', edgecolor='black',
                            width=2, headwidth=8))
ax.text(x0 + scalebar_len/2, y0 + 12000,
        'N', ha='center', va='bottom',
        fontsize=11, fontweight='bold')

# -------------------- Colorbar --------------------
cax = inset_axes(ax, width="20%", height="2%", loc="upper right",
                 bbox_to_anchor=(-0.05, -0.15, 1, 1),
                 bbox_transform=ax.transAxes, borderpad=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
plt.colorbar(sm, cax=cax, orientation="horizontal").ax.tick_params(labelsize=8)

# -------------------- Final Styling --------------------
ax.set_axis_off()
ax.margins(0)
fig.subplots_adjust(0, 0, 1, 1)

# -------------------- Save Figure --------------------
os.makedirs(os.path.join(script_dir, "results"), exist_ok=True)
plt.savefig(os.path.join(script_dir, "results/municipality_risk_map.png"),
            dpi=300, bbox_inches="tight", pad_inches=0)
