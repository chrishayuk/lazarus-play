#!/usr/bin/env python3
"""
Geometry of the Sydney→Canberra Flip
Visualisation from residual_trajectory, direction_angles, and subspace_decomposition.
Experiment: 5fd26ee8-63cc-4a9b-ba99-daa9a1de4bbc
Model: google/gemma-3-4b-it
Prompt: "The capital city of Australia is"
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_SYD   = "#e05c4b"   # Sydney — warm red
C_CAN   = "#3a7fc1"   # Canberra — blue
C_MEL   = "#a0a8b0"   # Melbourne — grey
C_ANNOT = "#2c3e50"   # annotation text
BG      = "white"

# ── Data: residual_trajectory ─────────────────────────────────────────────────
layers = list(range(34))

sydney_proj = [
    -2.1,   -3.7,   -2.2,   -2.7,   13.4,   18.1,   31.4,   57.0,
    98.6,  125.0,  158.9,  187.7,  223.1,  275.9,  290.0,  313.2,
   345.9,  398.0,  452.3,  538.0,  650.7,  895.8,  997.4, 1187.9,
  1847.9, 2726.3, 3282.8, 3396.2, 3837.5, 3785.6, 3801.6, 3119.2,
  2692.0, 2571.0,
]

canberra_proj = [
     9.2,   7.3,   9.0,   7.7,   4.5,   7.2,  15.5,  33.2,
    42.9,  61.1,  66.4,  72.5,  79.8,  87.8,  74.1,  88.9,
    83.8, 111.7,  37.0,  36.6,  54.2, 112.5, 138.1, 209.1,
   300.3, 344.7, 735.2, 1078.8, 1210.5, 1862.5, 1986.2, 2175.4,
  2406.5, 1983.7,
]

melbourne_proj = [
   -11.7,  -11.7,   -9.6,  -13.8,  -23.4,  -23.2,  -20.3,  -39.4,
   -83.0, -120.4, -153.6, -198.4, -221.7, -239.1, -247.5, -245.7,
  -242.0, -255.1, -277.5, -253.9, -215.8,  -20.1,   22.6,  246.6,
   751.1, 1464.2, 1792.1, 1879.8, 2108.1, 2015.3, 2041.2, 1296.0,
   768.2,  654.9,
]

# Sydney / Canberra competition ratio (layers where both > 0)
ratio_layers, ratio_vals = [], []
for l, (s, c) in enumerate(zip(sydney_proj, canberra_proj)):
    if s > 0 and c > 0:
        ratio_layers.append(l)
        ratio_vals.append(s / c)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.9, 1],
    hspace=0.45, wspace=0.38,
    left=0.07, right=0.97, top=0.92, bottom=0.07,
)
ax_traj  = fig.add_subplot(gs[0, :])
ax_ratio = fig.add_subplot(gs[1, 0])
ax_vocab = fig.add_subplot(gs[1, 1])
ax_dirs  = fig.add_subplot(gs[1, 2])

fig.suptitle(
    'The Sydney→Canberra Flip — Geometry of the L26 Decision\n'
    '"The capital city of Australia is"  ·  google/gemma-3-4b-it',
    fontsize=13, fontweight='bold', color=C_ANNOT, y=0.975,
)


# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Residual Trajectory
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_traj

# Shaded event regions
shading = [
    (23.5, 24.5, '#f39c12', 0.12),  # L24 Head 1
    (24.5, 25.5, '#e74c3c', 0.12),  # L25 FFN peak
    (25.5, 26.5, '#8e44ad', 0.22),  # L26 FFN bottleneck
    (28.5, 29.5, '#2980b9', 0.12),  # L29 Canberra surge
    (30.5, 32.5, '#27ae60', 0.08),  # L31-32 convergence zone
]
for x0, x1, col, alpha in shading:
    ax.axvspan(x0, x1, alpha=alpha, color=col, zorder=0, linewidth=0)

ax.axhline(0, color='#bdc3c7', linewidth=0.8, linestyle='--', zorder=1)

# Lines
ax.plot(layers, melbourne_proj, color=C_MEL, linewidth=1.6, zorder=2,
        label='Melbourne', alpha=0.85)
ax.plot(layers, canberra_proj,  color=C_CAN, linewidth=2.3, zorder=3,
        label='Canberra (correct answer)')
ax.plot(layers, sydney_proj,    color=C_SYD, linewidth=2.3, zorder=4,
        label='Sydney (misconception)')

# ── Annotations ──────────────────────────────────────────────────────────────
ann_kw = dict(fontsize=7.8, color=C_ANNOT,
              bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.88, ec='none'))
arr_kw = dict(arrowstyle='->', lw=0.85, color=C_ANNOT)

ax.annotate('L4\nSydney\nemerges', xy=(4, 13.4), xytext=(4, 380),
            ha='center', arrowprops=arr_kw, **ann_kw)
ax.annotate('L24\nHead 1\nfires', xy=(24, 1848), xytext=(21.5, 2700),
            ha='center', arrowprops=arr_kw, **ann_kw)
ax.annotate('L25 FFN\nSydney 92.2%\n(logit-lens peak)', xy=(25, 2726), xytext=(22.5, 3700),
            ha='center', arrowprops=arr_kw, **ann_kw)
ax.annotate('L26 FFN\nbottleneck\n★', xy=(26, 3283), xytext=(27.5, 4150),
            ha='center', arrowprops=dict(arrowstyle='->', lw=1.1, color='#8e44ad'),
            fontsize=8, color='#8e44ad',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0eaff', alpha=0.95, ec='#8e44ad', lw=0.8))
ax.annotate('L29\nCanberra\n+652 boost', xy=(29, 1862), xytext=(30.5, 900),
            ha='center', arrowprops=dict(arrowstyle='->', lw=0.85, color=C_CAN),
            fontsize=7.8, color=C_CAN,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.88, ec='none'))
ax.annotate('L33 — model outputs Canberra\n(Sydney raw proj still higher: 2571 vs 1984)',
            xy=(33, 2571), xytext=(29.5, 3400),
            ha='center', arrowprops=arr_kw,
            fontsize=7.8, color=C_ANNOT,
            bbox=dict(boxstyle='round,pad=0.3', fc='#fffbea', alpha=0.95, ec='#e0c060', lw=0.8))

# Callout box for the central finding
ax.text(
    14, 3500,
    '↑ Sydney raw projection leads at every layer,\n'
    '  including the final output layer.\n'
    '  The flip is mediated by layer norm.',
    fontsize=8.5, color='#555', style='italic',
    bbox=dict(boxstyle='round,pad=0.4', fc='#f9f9f9', alpha=0.9, ec='#ccc', lw=0.8),
)

ax.set_xlabel('Layer', fontsize=10)
ax.set_ylabel('Projection onto unembedding direction\n(dot product with residual stream)', fontsize=9)
ax.set_title('Residual Stream Projections Across All 34 Layers', fontsize=11, pad=8)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.set_xlim(-0.5, 33.5)
ax.set_ylim(-500, 4600)
ax.set_xticks(range(0, 34, 2))
ax.tick_params(labelsize=8)
ax.spines[['top', 'right']].set_visible(False)

# Small region labels along the x-axis
for lyr, label, col in [
    (24, 'Head 1', '#c0770a'),
    (25, 'L25 FFN', '#b03020'),
    (26, 'L26 FFN ★', '#6b2fa0'),
    (29, 'L29', '#1a6090'),
]:
    ax.text(lyr, -430, label, ha='center', va='bottom', fontsize=7, color=col, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Competition Ratio
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_ratio

ax.plot(ratio_layers, ratio_vals, color='#8e44ad', linewidth=2.2, zorder=3)
ax.fill_between(ratio_layers, 1, ratio_vals, alpha=0.13, color='#8e44ad', zorder=2)
ax.axhline(1, color='#bdc3c7', linewidth=1, linestyle='--', zorder=1)
ax.axvline(26, color='#8e44ad', linewidth=1.2, linestyle=':', alpha=0.6, zorder=2)

# Mark peak at L25
peak_i = ratio_vals.index(max(ratio_vals))
ax.annotate(f'L25 peak\n{max(ratio_vals):.1f}×', xy=(ratio_layers[peak_i], max(ratio_vals)),
            xytext=(18, max(ratio_vals) + 0.9),
            fontsize=8, color='#8e44ad',
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))

ax.annotate(f'L26\n{ratio_vals[ratio_layers.index(26)]:.1f}×',
            xy=(26, ratio_vals[ratio_layers.index(26)]),
            xytext=(28, 5.5),
            fontsize=8, color='#8e44ad',
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))

# Final value at L33
ax.annotate(f'L33: {ratio_vals[-1]:.2f}×\nmodel → Canberra',
            xy=(33, ratio_vals[-1]),
            xytext=(29.5, 3.2),
            fontsize=7.5, color=C_ANNOT,
            arrowprops=dict(arrowstyle='->', color=C_ANNOT, lw=0.8),
            bbox=dict(boxstyle='round,pad=0.25', fc='#fffbea', alpha=0.95, ec='#e0c060', lw=0.7))

ax.text(33.3, 1.12, '1×\nparity', ha='left', va='bottom', fontsize=7, color='#95a5a6')
ax.set_xlabel('Layer', fontsize=9)
ax.set_ylabel('Sydney proj / Canberra proj', fontsize=8.5)
ax.set_title('Competition Ratio\n(raw dot products)', fontsize=10)
ax.set_xlim(4, 33.5)
ax.set_ylim(0.5, 11)
ax.tick_params(labelsize=8)
ax.spines[['top', 'right']].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Vocabulary Space Compass
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_vocab
ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.55, 1.55)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Vocabulary Space Geometry\n(pairwise unembedding angles)', fontsize=10)

# Background circle
circle = plt.Circle((0, 0), 1.0, fill=False, color='#dfe6e9', linewidth=1.0)
ax.add_patch(circle)

# Origin dot
ax.plot(0, 0, 'o', color='#636e72', markersize=4, zorder=5)

# City directions (Sydney as reference axis at 0°)
cities = [
    ('Sydney',    0.0,   C_SYD,  2.6, 'bold'),
    ('Melbourne', 52.2,  C_MEL,  1.8, 'normal'),
    ('Brisbane',  67.8,  '#b0b8c0', 1.3, 'normal'),
    ('Perth',     84.2,  '#c8ced4', 1.1, 'normal'),
    ('Canberra',  85.0,  C_CAN,  2.6, 'bold'),
]

for name, deg, col, lw, fw in cities:
    rad = np.radians(deg)
    dx, dy = np.cos(rad), np.sin(rad)
    ax.annotate('', xy=(dx * 0.95, dy * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw))
    off = 1.22
    ax.text(dx * off, dy * off, name,
            ha='center', va='center', fontsize=8.5, color=col, fontweight=fw)

# Arc: Sydney–Canberra (85°)
arc_sc = Arc((0, 0), 0.52, 0.52, angle=0, theta1=0, theta2=85,
             color='#8e44ad', lw=1.3, linestyle='--')
ax.add_patch(arc_sc)
ax.text(0.36, 0.26, '85°', fontsize=8.5, color='#8e44ad', ha='center', fontweight='bold')

# Arc: Sydney–Melbourne (52°)
arc_sm = Arc((0, 0), 0.34, 0.34, angle=0, theta1=0, theta2=52,
             color=C_MEL, lw=1.0, linestyle=':')
ax.add_patch(arc_sm)
ax.text(0.22, 0.09, '52°', fontsize=7.5, color='#7f8c8d', ha='center')

# Cluster brace label
ax.text(0.68, -0.14, 'large-city\ncluster', ha='center', va='top', fontsize=7.5,
        color='#7f8c8d', style='italic')
# Bracket spanning the cluster
ax.annotate('', xy=(np.cos(np.radians(67.8)) * 1.08, np.sin(np.radians(67.8)) * 1.08),
            xytext=(np.cos(0) * 1.08, 0),
            arrowprops=dict(arrowstyle=']-[', color='#b0b8c0', lw=0.8))

# Subspace fraction note
ax.text(0, -1.45,
        '<0.6% of 2560D is "about cities"\n99.4% of residual is orthogonal to all city directions',
        ha='center', va='center', fontsize=7.5, color='#636e72', style='italic')


# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Direction Angles at Layer 26
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_dirs

groups  = ['Neuron 9444', 'Neuron 9182', 'FFN output']
g_angles_syd = [90.53, 104.84, 86.52]
g_angles_can = [93.37,  90.26, 86.05]

x    = np.arange(len(groups))
w    = 0.32
bars_syd = ax.bar(x - w/2, g_angles_syd, width=w, color=C_SYD, alpha=0.80,
                  label='angle to Sydney',   edgecolor='white', linewidth=0.5, zorder=3)
bars_can = ax.bar(x + w/2, g_angles_can, width=w, color=C_CAN, alpha=0.80,
                  label='angle to Canberra', edgecolor='white', linewidth=0.5, zorder=3)

# 90° orthogonal reference
ax.axhline(90, color='#c0392b', linewidth=1.3, linestyle='--', zorder=4, alpha=0.75)
ax.text(2.62, 90.4, '90° = orthogonal', ha='right', va='bottom',
        fontsize=7.5, color='#c0392b', alpha=0.85)

# Shade zones
ax.axhspan(84, 90, alpha=0.06, color=C_CAN,  zorder=0)   # <90 → slightly pro
ax.axhspan(90, 107, alpha=0.06, color=C_SYD, zorder=0)   # >90 → slightly anti
ax.text(-0.55, 87.2, '< 90°\npro', fontsize=7, color=C_CAN, ha='left', style='italic')
ax.text(-0.55, 92.5, '> 90°\nanti', fontsize=7, color=C_SYD, ha='left', style='italic')

# Value labels
for bar_group in [bars_syd, bars_can]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                f'{h:.1f}°', ha='center', va='bottom', fontsize=8, color=C_ANNOT)

# Dot product annotation for FFN output
ax.text(2, 84.5, 'FFN dot Canberra: 344.9\nFFN dot Sydney:   294.1',
        ha='center', va='bottom', fontsize=7.2, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f0f4ff', alpha=0.9, ec='#aac', lw=0.7))

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=8.5)
ax.set_ylabel('Angle (degrees)', fontsize=9)
ax.set_title('Direction Angles at Layer 26\n(neuron down-proj columns & FFN output)', fontsize=10)
ax.set_ylim(84, 108)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.spines[['top', 'right']].set_visible(False)


# ── Save ─────────────────────────────────────────────────────────────────────
out_png = 'geometry_sydney_canberra.png'
out_pdf = 'geometry_sydney_canberra.pdf'
plt.savefig(out_png, dpi=160, bbox_inches='tight', facecolor=BG)
plt.savefig(out_pdf, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out_png}  {out_pdf}')
plt.close()
