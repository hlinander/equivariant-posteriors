import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

VIEW_LON = 20.0
VIEW_LAT = 20.0

def sph_to_cart(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])

view = sph_to_cart(VIEW_LON, VIEW_LAT)
up_world = np.array([0, 0, 1.0])
right = np.cross(up_world, view); right /= np.linalg.norm(right)
up = np.cross(view, right); up /= np.linalg.norm(up)

def project(vecs):
    vecs = np.atleast_2d(vecs)
    return vecs @ right, vecs @ up, vecs @ view

def vec_from_thetaphi(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)

def clip_to_front(u, v, depth, threshold=0.0):
    """Sutherland-Hodgman clip: retain only depth >= threshold, interpolating at the limb."""
    out_u, out_v = [], []
    n = len(u)
    for i in range(n):
        j = (i + 1) % n
        d0, d1 = depth[i], depth[j]
        inside0 = d0 >= threshold
        inside1 = d1 >= threshold
        if inside0:
            out_u.append(u[i])
            out_v.append(v[i])
        if inside0 != inside1:
            t = (threshold - d0) / (d1 - d0)
            out_u.append(u[i] + t * (u[j] - u[i]))
            out_v.append(v[i] + t * (v[j] - v[i]))
    return np.array(out_u), np.array(out_v)

def edge_segs_clipped(u, v, depth):
    """Edge-by-edge clip to depth >= 0, returning individual line segments.

    Unlike Sutherland-Hodgman polygon clipping, this never connects two limb
    intersection points, so no chord appears where the boundary dips behind
    the sphere.
    """
    segs = []
    n = len(u)
    for i in range(n):
        j = (i + 1) % n
        d0, d1 = depth[i], depth[j]
        if d0 < 0 and d1 < 0:
            continue
        if d0 >= 0 and d1 >= 0:
            segs.append([(u[i], v[i]), (u[j], v[j])])
        else:
            t = -d0 / (d1 - d0)
            ui = u[i] + t * (u[j] - u[i])
            vi = v[i] + t * (v[j] - v[i])
            if d0 >= 0:
                segs.append([(u[i], v[i]), (ui, vi)])
            else:
                segs.append([(ui, vi), (u[j], v[j])])
    return segs

def draw_base_pixel_outlines(ax, linewidth=1.4, color='black'):
    """Overlay the 12 nside=1 base-pixel boundaries as thick lines on any panel."""
    segs = []
    for ipix in range(12):
        verts = hp.boundaries(1, ipix, step=80, nest=False).T
        u, v, depth = project(verts)
        if np.max(depth) < 0:
            continue
        segs.extend(edge_segs_clipped(u, v, depth))
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=linewidth, zorder=6))

def draw_panel(ax, nside, title):
    npix = hp.nside2npix(nside)

    ipix_all = np.arange(npix)
    ipix_nest = hp.ring2nest(nside, ipix_all)
    base_pixel = ipix_nest // (nside * nside)

    light_base = 0
    dark_base = 4

    step = max(8, 40 // nside)

    grid_segs = []
    for ipix in range(npix):
        verts = hp.boundaries(nside, ipix, step=step, nest=False).T
        u, v, depth = project(verts)

        if np.max(depth) < 0.0:
            continue

        bp = base_pixel[ipix]
        # ── color options: uncomment one pair ──────────────────────────────
        # fc_light, fc_dark = (0.78, 0.78, 0.78), (0.45, 0.45, 0.45)  # neutral grey
        # fc_light, fc_dark = (0.82, 0.79, 0.76), (0.55, 0.50, 0.46)  # warm grey
        fc_light, fc_dark   = (0.76, 0.78, 0.82), (0.46, 0.50, 0.57)  # soft blue-grey (subtle)
        # fc_light, fc_dark = (0.74, 0.80, 0.86), (0.40, 0.52, 0.66)  # steel blue-grey
        # fc_light, fc_dark = (0.78, 0.81, 0.72), (0.50, 0.55, 0.40)  # sage
        # fc_light, fc_dark = (0.85, 0.82, 0.74), (0.58, 0.52, 0.40)  # sand / warm tan
        # fc_light, fc_dark = (0.80, 0.75, 0.82), (0.52, 0.42, 0.58)  # dusty mauve
        # fc_light, fc_dark = (0.72, 0.82, 0.80), (0.38, 0.58, 0.55)  # muted teal
        # ───────────────────────────────────────────────────────────────
        if bp == light_base:
            fc = fc_light
        elif bp == dark_base:
            fc = fc_dark
        else:
            fc = 'none'

        cu, cv = clip_to_front(u, v, depth)
        if len(cu) < 3:
            continue

        # Fill without edge so the clipped-polygon chord is never drawn
        ax.fill(cu, cv, facecolor=fc if fc != 'none' else 'none',
                edgecolor='none', zorder=1)

        # Collect edge segments via edge-by-edge clipping (no chord artifact)
        grid_segs.extend(edge_segs_clipped(u, v, depth))

    if grid_segs:
        ax.add_collection(LineCollection(grid_segs, colors='black', linewidths=0.4, zorder=2))

    # Pixel centers
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    centers = vec_from_thetaphi(theta, phi)
    cu, cv, cdepth = project(centers)
    front = cdepth > 0.02
    msize = {1: 70, 2: 35, 4: 12, 8: 6}[nside]
    ax.scatter(cu[front], cv[front], s=msize, c='black', zorder=5)

    draw_base_pixel_outlines(ax)

    circ = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(circ), np.sin(circ), color='black', linewidth=0.8, zorder=2)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
configs = [
    (axes[0, 0], 1, "Nside=1, Npix=12"),
    (axes[0, 1], 2, "Nside=2, Npix=48"),
    (axes[1, 0], 8, "Nside=8, Npix=768"),
    (axes[1, 1], 4, "Nside=4, Npix=192"),
]
for ax, nside, title in configs:
    draw_panel(ax, nside, title)

plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
out = 'healpix.png'
plt.savefig(out, dpi=600, bbox_inches='tight')
plt.show()
print(f"Saved: {out}")
