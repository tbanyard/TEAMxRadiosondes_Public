import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import datetime

colors = [
    "#fff09d",
    "#eab800",
    "#ffd1ff",
    "#d35cd3",
    "#4dffa9",
    "#008746",
    "#a3e3ff",
    "#008bc7"
]

labels = [
    "2025-02-26 08:00 (A)",
    "2025-02-26 08:00 (B)",
    "2025-02-26 11:00 (A)",
    "2025-02-26 11:00 (B)",
    "2025-02-26 14:00 (A)",
    "2025-02-26 14:00 (B)",
    "2025-02-26 17:00 (A)",
    "2025-02-26 17:00 (B)"
]

# Create figure
fig, ax = plt.subplots(figsize=(2, 2))
ax.axis("off")   # blank canvas

# Vertical placement controls
y_start = 0.92
dy = 0.12
line_x1 = 0.08
line_x2 = 0.38

for i, (col, text) in enumerate(zip(colors, labels)):
    y = y_start - i * dy

    # short legend line
    ax.plot([line_x1, line_x2], [y, y], lw=4, color=col, transform=ax.transAxes, clip_on=False)

    # label text
    ax.text(line_x2 + 0.05, y, text, transform=ax.transAxes,
            va="center", ha="left", fontsize=10)

# Add dashed bounding box
pad_x = 0.1
pad_y = 0.1

box_x = line_x1 - pad_x
box_y = (y_start - (len(labels)-1)*dy) - pad_y
box_w = 1.48
box_h = (len(labels)-1)*dy + 2*pad_y

white_box = FancyBboxPatch(
    (box_x, box_y),
    box_w, box_h,
    boxstyle="square,pad=0",
    linewidth=0,
    facecolor="white",
    transform=ax.transAxes,
    clip_on=False,
    zorder=0,       # behind everything
)
ax.add_patch(white_box)

bbox = FancyBboxPatch(
    (box_x, box_y),
    box_w, box_h,
    boxstyle="square,pad=0",
    linewidth=1.2,
    edgecolor="black",
    linestyle="--",
    facecolor="none",
    transform=ax.transAxes,
    clip_on=False
)
ax.add_patch(bbox)

# Ensure the 'plots' subdirectory exists (create if needed)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
figname = f"legend_{timestamp}.png"
os.makedirs("plots", exist_ok=True)
plot_path = os.path.join("plots", figname)
plt.savefig(plot_path, dpi=450, bbox_inches="tight", transparent=True)
print(f"Saved figure {figname} in directory {os.path.join(os.getcwd(), 'plots')}")
plt.close()