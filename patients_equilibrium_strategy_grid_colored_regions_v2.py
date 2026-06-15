import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ============================================================
# Parameters
# ============================================================
Cw = 0.5
delta1 = 0.5
delta2 = 0.4
p2 = 0.3
V = 2.3
T = 0.5
s = 0.5
T1 = T + p2
R2 = V - T - p2
p1 = 0.5
R1 = V - p1
T2 = p1
Lambda = 6
LambdaE = 1
r_on = p1 + 0.2
r_off = p2 + 0.2
r_leave = 0.5

rightSide1 = R1 - delta1 * T1 + delta1 * s
rightSide2 = R2 - delta2 * T2 + s

# ============================================================
# Grid
# ============================================================
x1 = np.linspace(0.1, 8, 3000)
x2 = np.linspace(0.1, 8, 3000)
X1, X2 = np.meshgrid(x1, x2)  # x-axis: mu_off, y-axis: mu_on

A = (R2 + s - delta2 * T2) * delta2
B = (R2 + s - delta2 * T2) * (X2 + delta2 * X1) - 2 * delta2 * Cw
D = (R2 + s - delta2 * T2) * (X2 - delta2 * X1)
sqrt_term = np.sqrt(D**2 + 4 * delta2**2 * Cw**2)

A2 = (1 - delta1) * T1 - (1 - delta2) * T2 - (1 - delta1) * s
B2 = 2 * (1 - delta2) * Cw + (1 - delta2) * ((T1 - s) - (1 - delta2) / (1 - delta1) * T2) * (X1 - Lambda) \
     - ((1 - delta1) * (T1 - s) - (1 - delta2) * T2) * (X2 - delta2 * Lambda)
D2 = ((1 - delta2) * (T1 - s) - (1 - delta2) * (1 - delta2) / (1 - delta1) * T2) \
     * (X1 - Lambda + (1 - delta1) / (1 - delta2) * (X2 - delta2 * Lambda))

A1 = (R1 + delta1 * s - delta1 * T1) * delta1
B1 = (R1 + delta1 * s - delta1 * T1) * (X1 + delta1 * X2) - 2 * delta1 * Cw
D1 = (R1 + delta1 * s - delta1 * T1) * (X1 - delta1 * X2)

sqrt_term1 = np.sqrt(D1**2 + 4 * delta1**2 * Cw**2)
sqrt_term2 = np.sqrt(D2**2 + 4 * (1 - delta2)**2 * Cw**2)

lambda_off = (B - sqrt_term) / (2 * A)
lambda_on = (B1 - sqrt_term1) / (2 * A1)
lambda_on2 = (-B2 + sqrt_term2) / (2 * A2) / (1 - delta2)

# ============================================================
# Curves
# ============================================================
F1 = Cw * (1 / X2 + delta1 / X1) - rightSide1
F2 = Cw * (1 / X1) + delta2 * Cw * (1 / X2) - rightSide2
F3 = (X2 - delta2 * X1) / (1 - delta2 * delta1) + (delta2 * Cw) / ((R2 + s - delta2 * T2) - delta2 * (R1 - delta1 * T1 + delta1 * s)) \
     - Cw / ((R1 - delta1 * T1) - delta1 * (R2 - delta2 * T2))
F4 = (X1 - delta1 * X2) / (1 - delta2 * delta1) + (delta1 * Cw) / ((R1 - delta1 * T1) - delta1 * (R2 - delta2 * T2)) \
     - Cw / ((R2 + s - delta2 * T2) - delta2 * (R1 - delta1 * T1 + delta1 * s))
F5 = ((1 - delta1) * X2 + (1 - delta2) * X1) / (1 - delta2 * delta1) - Lambda \
     - ((1 - delta2) * Cw) / ((R2 + s - delta2 * T2) - delta2 * (R1 - delta1 * T1 + delta1 * s)) \
     - ((1 - delta1) * Cw) / ((R1 - delta1 * T1) - delta1 * (R2 - delta2 * T2))
F6 = Cw * (1 / (X2 - Lambda)) + delta1 * Cw * (1 / (X1 - delta1 * Lambda)) - rightSide1
F7 = Cw * (1 / (X1 - Lambda)) + delta2 * Cw * (1 / (X2 - delta2 * Lambda)) - rightSide2
F8 = (1 - delta1) * Cw * (1 / (X1 - Lambda)) - (1 - delta2) * Cw * (1 / (X2 - delta2 * Lambda)) \
     - ((1 - delta2) * T2 + (1 - delta1) * s - (1 - delta1) * T1)
F9 = (1 - delta2) * Cw * (1 / (X2 - Lambda)) - (1 - delta1) * Cw * (1 / (X1 - delta1 * Lambda)) \
     - ((1 - delta1) * T1 - (1 - delta2) * T2 - (1 - delta1) * s)

F10 = rightSide1 - Cw / (X2 - delta2 * lambda_off) - delta1 * Cw / (X1 - lambda_off)
F11 = rightSide2 - Cw / (X1 - delta1 * lambda_on) - delta2 * Cw / (X2 - lambda_on)
F12 = rightSide1 - Cw / (X2 - delta2 * (Lambda - lambda_on2) - lambda_on2) \
      - delta1 * Cw / (X1 - (Lambda - lambda_on2) - delta1 * lambda_on2)

# ============================================================
# Original masking rules for the curves
# ============================================================
threshold = -((Cw * (-1 + delta1 * delta2)) / (R2 + s - (R1 + T2 + s * delta1 - T1 * delta1) * delta2))

region1 = X1 <= threshold
region2 = X1 >= threshold
region3 = (threshold <= X1) & (X1 <= threshold + Lambda)
region4 = (threshold <= X1) & (X1 <= threshold + delta1 * Lambda)
region5 = (X1 >= threshold + delta1 * Lambda) & (X1 <= threshold + Lambda)
region6 = (X1 <= threshold + Lambda) \
          & (X2 >= (Cw - Cw * delta1 * delta2) / (R1 - delta1 * (R2 + T1 - T2 * delta2)) + Lambda) \
          & (X1 >= delta1 * Lambda + delta1 * Cw / rightSide1) \
          & (X2 >= Lambda + Cw / rightSide1)
region7 = (X1 >= threshold + Lambda) \
          & (X2 >= delta2 * Lambda) \
          & (X1 >= Lambda + Cw / rightSide2) \
          & (X2 >= delta2 * Lambda + delta2 * Cw / rightSide2)
region8 = (X1 >= threshold + Lambda) \
          & (X1 > Lambda) \
          & (X2 > delta2 * Lambda) \
          & (X1 >= Lambda + Cw / rightSide2) \
          & (X2 >= delta2 * Lambda + delta1 * Cw / rightSide2)
region9 = (X1 >= threshold + delta1 * Lambda) \
          & (X1 > delta1 * Lambda) \
          & (X2 > Lambda)
region10 = (threshold <= X1) & (X1 <= threshold + Lambda)
region11 = X1 > 0

F1_masked = np.ma.masked_where(~region1, F1)
F2_masked = np.ma.masked_where(~region2, F2)
F3_masked = np.ma.masked_where(~region3, F3)
F4_masked = np.ma.masked_where(~region4, F4)
F5_masked = np.ma.masked_where(~region5, F5)
F6_masked = np.ma.masked_where(~region6, F6)
F7_masked = np.ma.masked_where(~region7, F7)
F8_masked = np.ma.masked_where(~region8, F8)
F9_masked = np.ma.masked_where(~region9, F9)
F10_masked = np.ma.masked_where(~region10, F10)
F11_masked = np.ma.masked_where(~region11, F11)
F12_masked = np.ma.masked_where(~region11, F12)

# ============================================================
# Helpers
# ============================================================
def contour_vertices(cs):
    verts = []
    for seg_group in cs.allsegs:
        for seg in seg_group:
            if len(seg) > 0:
                verts.append(seg)
    return verts


def closest_intersection(cs1, cs2):
    best_d = np.inf
    best_p = None
    for s1 in contour_vertices(cs1):
        for s2 in contour_vertices(cs2):
            diff = s1[:, None, :] - s2[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            idx = np.unravel_index(np.argmin(d2), d2.shape)
            d = d2[idx]
            if d < best_d:
                best_d = d
                best_p = (s1[idx[0]] + s2[idx[1]]) / 2.0
    return best_p


def draw_guides(ax, point, x0=0.1, y0=0.1):
    if point is None:
        return
    x, y = point
    ax.plot([x0, x], [y, y], color='0.45', lw=1.2, linestyle=(0, (5, 4)), zorder=3)
    ax.plot([x, x], [y0, y], color='0.45', lw=1.2, linestyle=(0, (5, 4)), zorder=3)


def data_to_pixel(x, y, xmin, xmax, ymin, ymax, width, height):
    c = int(round((x - xmin) / (xmax - xmin) * (width - 1)))
    r = int(round((ymax - y) / (ymax - ymin) * (height - 1)))
    return r, c


def flood_fill_regions(boundary_mask, seeds, xmin, xmax, ymin, ymax):
    h, w = boundary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    region_pixels = {}
    for name, (x, y) in seeds.items():
        sr, sc = data_to_pixel(x, y, xmin, xmax, ymin, ymax, w, h)
        q = deque([(sr, sc)])
        pixels = []
        while q:
            r, c = q.popleft()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if boundary_mask[r, c] or visited[r, c]:
                continue
            visited[r, c] = True
            pixels.append((r, c))
            q.append((r + 1, c))
            q.append((r - 1, c))
            q.append((r, c + 1))
            q.append((r, c - 1))
        region_pixels[name] = pixels
    return region_pixels


# ============================================================
# First pass: get clean intersection points from contours
# ============================================================
fig_tmp, ax_tmp = plt.subplots(figsize=(6, 6))
ax_tmp.set_xlim(0.1, 8)
ax_tmp.set_ylim(0.1, 8)
ax_tmp.axis('off')

c1 = ax_tmp.contour(X1, X2, F1_masked, levels=[0], colors='black', linewidths=1)
c2 = ax_tmp.contour(X1, X2, F2_masked, levels=[0], colors='black', linewidths=1)
c3 = ax_tmp.contour(X1, X2, F3_masked, levels=[0], colors='black', linewidths=1)
c4 = ax_tmp.contour(X1, X2, F4_masked, levels=[0], colors='black', linewidths=1)
c5 = ax_tmp.contour(X1, X2, F5_masked, levels=[0], colors='black', linewidths=1)
c6 = ax_tmp.contour(X1, X2, F6_masked, levels=[0], colors='black', linewidths=1)
c7 = ax_tmp.contour(X1, X2, F7_masked, levels=[0], colors='black', linewidths=1)
c8 = ax_tmp.contour(X1, X2, F8_masked, levels=[0], colors='black', linewidths=1)
c9 = ax_tmp.contour(X1, X2, F9_masked, levels=[0], colors='black', linewidths=1)
c10 = ax_tmp.contour(X1, X2, F10_masked, levels=[0], colors='black', linewidths=1)

p_bottom = closest_intersection(c1, c2)
p_top = closest_intersection(c4, c5)
p_right = closest_intersection(c3, c5)

plt.close(fig_tmp)

# ============================================================
# Hidden raster pass for region coloring
# Seal the three multi-curve intersection points so that B and V
# do not leak into neighboring regions.
# ============================================================
xmin, xmax = 0.1, 8
ymin, ymax = 0.1, 8
width, height = 1400, 1400
dpi = 200

fig_hidden = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
ax_hidden = fig_hidden.add_axes([0, 0, 1, 1])
ax_hidden.set_xlim(xmin, xmax)
ax_hidden.set_ylim(ymin, ymax)
ax_hidden.axis('off')

for Fm in [F1_masked, F2_masked, F3_masked, F4_masked, F5_masked,
           F6_masked, F7_masked, F8_masked, F9_masked, F10_masked]:
    ax_hidden.contour(X1, X2, Fm, levels=[0], colors='black', linewidths=6, antialiased=False)

# add outer frame for the flood fill domain
ax_hidden.plot([xmin, xmax, xmax, xmin, xmin],
               [ymin, ymin, ymax, ymax, ymin],
               color='black', lw=6, solid_capstyle='butt')

# seal the three key intersections in the hidden pass only
seal_pts = np.array([p_bottom, p_top, p_right])
ax_hidden.scatter(seal_pts[:, 0], seal_pts[:, 1], s=170, c='black', zorder=10)

canvas = FigureCanvasAgg(fig_hidden)
canvas.draw()
hidden_img = np.asarray(canvas.buffer_rgba())
plt.close(fig_hidden)

boundary_mask = hidden_img[..., 0] < 240

region_seeds = {
    'B': (0.30, 0.30),
    'BV': (1.55, 5.00),
    'V': (3.47, 7.70),
    'VF': (5.40, 6.10),
    'BVF': (3.70, 3.60),
    'F': (7.25, 3.25),
    'BF': (6.05, 1.40),
}

region_pixels = flood_fill_regions(boundary_mask, region_seeds, xmin, xmax, ymin, ymax)

region_colors = {
    'B':   '#f4cccc',
    'BV':  '#cfe2f3',
    'V':   '#d9ead3',
    'VF':  '#fff2cc',
    'BVF': '#ead1dc',
    'F':   '#d0e0e3',
    'BF':  '#fce5cd',
}

rgba = np.zeros((height, width, 4), dtype=np.uint8)
for name, pixels in region_pixels.items():
    rr = np.array([p[0] for p in pixels], dtype=int)
    cc = np.array([p[1] for p in pixels], dtype=int)
    color = plt.matplotlib.colors.to_rgba(region_colors[name], alpha=0.72)
    rgba[rr, cc] = (np.array(color) * 255).astype(np.uint8)

# ============================================================
# Final plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

# colored regions
ax.imshow(rgba, extent=[xmin, xmax, ymin, ymax], origin='upper', interpolation='nearest', zorder=0)

# black boundaries (visible pass; no sealing dots here)
for Fm in [F1_masked, F2_masked, F3_masked, F4_masked, F5_masked,
           F6_masked, F7_masked, F8_masked, F9_masked, F10_masked]:
    ax.contour(X1, X2, Fm, levels=[0], colors='black', linewidths=2.8, zorder=4)

# guides
for p in [p_top, p_right]:
    draw_guides(ax, p)

# labels (adjusted so B and V are inside their own regions)
labels = {
    'B':   (0.31, 0.31),
    'BV':  (1.70, 5.85),
    'V':   (3.48, 7.72),
    'VF':  (5.52, 6.15),
    'BVF': (3.70, 3.62),
    'F':   (7.28, 3.30),
    'BF':  (6.05, 1.45),
}
label_sizes = {
    'B': 17,
    'V': 17,
    'BV': 18,
    'VF': 18,
    'BVF': 18,
    'F': 18,
    'BF': 18,
}
for txt, (x, y) in labels.items():
    ax.text(
        x, y, txt,
        fontsize=label_sizes[txt],
        fontweight='bold',
        ha='center', va='center', family='serif',
        bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='none', alpha=0.62),
        zorder=6,
    )

ax.set_xlim(0.1, 8)
ax.set_ylim(0.1, 8)
ax.set_xlabel(r'$\mu_{\mathrm{off}}$', fontsize=18)
ax.set_ylabel(r'$\mu_{\mathrm{on}}$', fontsize=18)

# Hide numeric tick labels while keeping the grid
# and axis titles.
tick_positions = np.arange(1, 9, 1)
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both', length=0)
ax.grid(True, alpha=0.35, linewidth=0.8)

plt.tight_layout()

out_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
fig.savefig(out_dir / 'patients_equilibrium_strategy_grid_colored_regions_v2.png', dpi=300)
fig.savefig(out_dir / 'patients_equilibrium_strategy_grid_colored_regions_v2.pdf')
fig.savefig(out_dir / 'patients_equilibrium_strategy_grid_colored_regions_v2.svg')

plt.show()
