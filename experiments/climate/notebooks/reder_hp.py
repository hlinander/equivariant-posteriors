"""
Thesis cover — plasma palette, green outlines on the 12 base HEALPix pixels,
with a subtle land mask so the sphere reads as Earth (weather forecasting).

- Field: zonally symmetric (function of latitude only), SO(2)-invariant.
- Fine HEALPix grid (Nside=32) drawn subtly so it doesn't compete.
- HEALPix pixels whose centers lie over land are slightly darkened, so the
  continents are recognizable without breaking the SO(2) gradient story.
- The 12 base pixels (Nside=1) are outlined boldly in vivid green.
- A curved arrow traces a high-latitude ring to indicate rotation
  about the polar axis.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from global_land_mask import globe

# ---------- Camera ----------
VIEW_LON, VIEW_LAT = 25.0, 22.0
NSIDE = 64

def sph_to_cart(lo_d, la_d):
    lo, la = np.radians(lo_d), np.radians(la_d)
    return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])

view = sph_to_cart(VIEW_LON, VIEW_LAT)
up_world = np.array([0, 0, 1.0])
right = np.cross(up_world, view); right /= np.linalg.norm(right)
up_cam = np.cross(view, right); up_cam /= np.linalg.norm(up_cam)

def project(vecs):
    vecs = np.atleast_2d(vecs)
    return vecs @ right, vecs @ up_cam, vecs @ view

# ---------- Zonally symmetric field ----------
def zonal_field(nside):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lat = np.pi / 2 - theta
    return np.cos(2 * lat) + 0.18 * np.cos(4 * lat)

# ---------- Fine HEALPix pixels with subtle grid + land darkening ----------
def draw_pixels(ax, nside, field, cmap, vmax,
                grid_color=(0, 0, 0, 0.20), grid_lw=0.30,
                land_darken=0.30):
    """`land_darken` in [0,1]: 0 = no effect, 1 = full black for land pixels."""
    norm = Normalize(-vmax, vmax)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lat_deg = np.degrees(np.pi / 2 - theta)
    lon_deg = ((np.degrees(phi) + 180) % 360) - 180  # to [-180, 180]
    is_land = globe.is_land(lat_deg, lon_deg)

    for ipix in range(npix):
        verts = hp.boundaries(nside, ipix, step=max(10, 48 // nside)).T
        u, v, d = project(verts)
        if np.mean(d > 0) < 0.55:
            continue
        rgba = list(cmap(norm(field[ipix])))
        if is_land[ipix]:
            for k in range(3):
                rgba[k] *= (1.0 - land_darken)
        ax.fill(u, v, facecolor=tuple(rgba),
                edgecolor=grid_color, linewidth=grid_lw, zorder=2)

# ---------- 12 base pixels in bold green ----------
def draw_base_pixels(ax, color='#1fff55', linewidth=2.8, halo=True):
    for ipix in range(12):
        verts = hp.boundaries(1, ipix, step=80).T
        verts = np.concatenate([verts, verts[:1]], axis=0)  # close loop
        u, v, d = project(verts)
        if np.max(d) < 0:
            continue
        front = d > 0
        u_p = np.where(front, u, np.nan)
        v_p = np.where(front, v, np.nan)
        if halo:
            # Soft dark halo so green pops against bright field regions
            ax.plot(u_p, v_p, color=(0, 0, 0, 0.55),
                    linewidth=linewidth + 1.6, zorder=6,
                    solid_capstyle='round', solid_joinstyle='round')
        ax.plot(u_p, v_p, color=color, linewidth=linewidth, zorder=7,
                solid_capstyle='round', solid_joinstyle='round')

# ---------- Rotation arrow (follows a circle of constant latitude) ----------
def draw_rotation_arrow(ax, lat_deg=62, phi_start_deg=-50, phi_end_deg=165,
                        color='black', linewidth=2.6, halo=True):
    """Curved arrow tracing a latitude ring — visualizes rotation about the polar axis."""
    th = np.pi / 2 - np.radians(lat_deg)
    z = np.cos(th); r = np.sin(th)
    phis = np.linspace(np.radians(phi_start_deg),
                       np.radians(phi_end_deg), 240)
    pts = np.stack([r * np.cos(phis), r * np.sin(phis),
                    np.full_like(phis, z)], axis=-1)
    u, v, d = project(pts)
    front_idx = np.where(d > 0)[0]
    if len(front_idx) < 4:
        return
    splits = np.where(np.diff(front_idx) > 1)[0]
    if len(splits):
        front_idx = front_idx[: splits[0] + 1]
    u_v, v_v = u[front_idx], v[front_idx]

    if halo:
        ax.plot(u_v[:-2], v_v[:-2], color=(1, 1, 1, 0.6),
                linewidth=linewidth + 0.9, zorder=8,
                solid_capstyle='round')
    ax.plot(u_v[:-2], v_v[:-2], color=color, linewidth=linewidth,
            zorder=9, solid_capstyle='round')

    from matplotlib.patches import FancyArrowPatch
    if halo:
        halo_arrow = FancyArrowPatch((u_v[-3], v_v[-3]), (u_v[-1], v_v[-1]),
                                     arrowstyle='-|>', mutation_scale=22,
                                     color=(1, 1, 1, 0.6),
                                     linewidth=linewidth + 0.9, zorder=8)
        ax.add_patch(halo_arrow)
    arrow = FancyArrowPatch((u_v[-3], v_v[-3]), (u_v[-1], v_v[-1]),
                            arrowstyle='-|>', mutation_scale=22,
                            color=color, linewidth=linewidth, zorder=9)
    ax.add_patch(arrow)

# ============================================================
#  Render
# ============================================================
field = zonal_field(NSIDE)
vmax = np.percentile(np.abs(field), 99)
cmap = plt.cm.plasma

fig, ax = plt.subplots(figsize=(8.5, 8.5), facecolor='white')
ax.set_facecolor('white')

draw_pixels(ax, NSIDE, field, cmap, vmax,
            grid_color=(0, 0, 0, 0.05), grid_lw=0.15,
            land_darken=0.22)
draw_base_pixels(ax, color='#1fff55', linewidth=2.8, halo=True)

# Sphere outline
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), color='black', linewidth=1.4, zorder=5)

draw_rotation_arrow(ax, lat_deg=62, phi_start_deg=-50, phi_end_deg=165,
                    color='black', linewidth=2.6, halo=True)

ax.set_xlim(-1.30, 1.30)
ax.set_ylim(-1.30, 1.30)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig('./cover_base.pdf', dpi=220,
            bbox_inches='tight', facecolor='white')
plt.close()
print("done")
