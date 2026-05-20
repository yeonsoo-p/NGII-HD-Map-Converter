import geopandas as gpd
import matplotlib.pyplot as plt

# ── Config ──
SHP_PATH = "B2_SURFACELINEMARK.shp"
ROUND_DIGITS = 2

# ── Load ──
gdf = gpd.read_file(SHP_PATH)
kind_col = [c for c in gdf.columns if c.lower() == "kind"][0]
center = gdf[gdf[kind_col].astype(str).str.strip() == "501"].copy()
center = center[center.geometry.notnull()].reset_index(drop=True)
geoms = list(center.geometry)
print(f"Centerline count: {len(center)}")

# ── Overlap detection (coordinate rounding) ──
coord_sets = []
for geom in geoms:
    pts = {(round(x, ROUND_DIGITS), round(y, ROUND_DIGITS)) for x, y, *_ in geom.coords}
    coord_sets.append(pts)

overlap_idx = set()
for i in range(len(coord_sets)):
    for j in range(i + 1, len(coord_sets)):
        shared = coord_sets[i] & coord_sets[j]
        if not shared:
            continue
        min_len = min(len(coord_sets[i]), len(coord_sets[j]))
        if len(shared) / min_len > 0.5:
            overlap_idx.add(i)
            overlap_idx.add(j)

print(f"Overlapping (red): {len(overlap_idx)}")

# ── Non-overlapping = offset double ──
non_ov_idx = [i for i in range(len(geoms)) if i not in overlap_idx]
print(f"Non-overlapping (blue): {len(non_ov_idx)}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(16, 16))

if non_ov_idx:
    center.iloc[non_ov_idx].plot(ax=ax, color="blue", linewidth=1.0, label=f"Offset double ({len(non_ov_idx)})")
if overlap_idx:
    center.iloc[list(overlap_idx)].plot(ax=ax, color="red", linewidth=1.2, label=f"Overlapping ({len(overlap_idx)})")

ax.set_aspect("equal")
ax.legend(fontsize=12)
ax.set_title("B2 Centerline (Kind=501)", fontsize=14)

# ── Scroll zoom + Drag pan ──
_drag = {"active": False, "x": None, "y": None}

def on_scroll(event):
    if event.inaxes != ax:
        return
    scale = 0.8 if event.button == "up" else 1.25
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xc, yc = event.xdata, event.ydata
    ax.set_xlim(xc - (xc - xlim[0]) * scale, xc + (xlim[1] - xc) * scale)
    ax.set_ylim(yc - (yc - ylim[0]) * scale, yc + (ylim[1] - yc) * scale)
    fig.canvas.draw_idle()

def on_press(event):
    if event.inaxes != ax or event.button != 1:
        return
    _drag.update(active=True, x=event.xdata, y=event.ydata)

def on_release(event):
    _drag["active"] = False

def on_motion(event):
    if not _drag["active"] or event.inaxes != ax:
        return
    dx, dy = _drag["x"] - event.xdata, _drag["y"] - event.ydata
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
    ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

plt.tight_layout()
plt.show()