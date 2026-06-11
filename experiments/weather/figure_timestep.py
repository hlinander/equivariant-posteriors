"""Recreate the one-timestep illustration figure for PEAR.

Two globes (surface field on HEALPix with green grid + translucent halo),
each with a volumetric pie cutout exposing the 13 upper-variable pressure
levels as nested shells, joined by a pear and an arrow:

    t_0   --(pear)-->   t_0 + dt

Run with:
    uv run --with pyvista python experiments/weather/figure_timestep.py
"""

import numpy as np
import healpy
import pyvista as pv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

HERE = Path(__file__).parent

NSIDE = 64
N_LEVELS = 13
RNG = np.random.default_rng(7)

# --- octant cutout / shell geometry --------------------------------------
WEDGE = 90.0  # octant: lon in [0, WEDGE], lat in [0, 90] removed
N_LAYERS = 1 + N_LEVELS  # surface + upper levels
R_OUT = 1.0
DR_SURFACE = 0.05
DR_LEVEL = 0.034
CMAP = "plasma"
CLIM = (-1.3, 2.2)


def synfast_nest(seed):
    ell = np.arange(3 * NSIDE)
    cl = 1.0 / (1.0 + ell.astype(float)) ** 3.0
    np.random.seed(seed)
    m = healpy.synfast(cl, NSIDE)
    m = healpy.reorder(m, r2n=True)
    return m / m.std()


class Fields:
    """Plausible weather-looking fields on the HEALPix grid.

    Static part (latitude gradient + land/topography from the real masks)
    plus smooth random 'weather' that drifts eastward with time.
    """

    def __init__(self):
        masks = np.load(HERE / "masks" / "masks_hp_64.npy")
        self.land = masks[0]
        self.topo = masks[2]
        self.noise_a = synfast_nest(3)
        self.noise_b = synfast_nest(11)

    def _pix(self, lon, lat):
        return healpy.ang2pix(NSIDE, lon % 360.0, lat, nest=True, lonlat=True)

    def sample(self, lon, lat, level, t):
        """level: 0 = surface, 1..N_LEVELS = upper levels. t: 0 or 1."""
        drift = 14.0 * t
        pix = self._pix(lon, lat)
        pix_adv = self._pix(lon - drift, lat)
        latw = np.cos(np.radians(lat)) ** 3
        if level == 0:
            static = -0.85 + 1.9 * latw + 0.18 * self.land[pix] - 0.15 * self.topo[pix]
            noise = 0.5 * self.noise_a[pix_adv]
        else:
            k = level - 1
            alpha = (k / max(N_LEVELS - 1, 1)) * np.pi / 2
            n = np.cos(alpha) * self.noise_a[pix_adv] + np.sin(alpha) * self.noise_b[pix_adv]
            static = -0.5 + (1.7 - 0.06 * k) * latw - 0.08 * k
            noise = 0.38 * n
        return static + noise


def lonlat_to_xyz(lon, lat, r):
    lonr, latr = np.radians(lon), np.radians(lat)
    return np.stack(
        [
            r * np.cos(latr) * np.cos(lonr),
            r * np.cos(latr) * np.sin(lonr),
            r * np.sin(latr),
        ],
        axis=-1,
    )


def grid_mesh(lon2d, lat2d, r2d, scalars):
    xyz = lonlat_to_xyz(lon2d, lat2d, r2d)
    grid = pv.StructuredGrid(
        np.ascontiguousarray(xyz[..., 0]),
        np.ascontiguousarray(xyz[..., 1]),
        np.ascontiguousarray(xyz[..., 2]),
    )
    grid["field"] = scalars.ravel(order="F")
    return grid


def layer_radii(layer):
    """Outer/inner radius of layer (0 = surface, 1.. = upper levels)."""
    if layer == 0:
        return R_OUT, R_OUT - DR_SURFACE
    r_out = R_OUT - DR_SURFACE - (layer - 1) * DR_LEVEL
    return r_out, r_out - DR_LEVEL


R_CORE = layer_radii(N_LAYERS - 1)[1]


def sphere_patch(fields, t, level, r, lat_range, lon_range, n=6):
    lat = np.linspace(*lat_range, int(abs(lat_range[1] - lat_range[0]) * 1.4) + 2)
    lon = np.linspace(*lon_range, int(abs(lon_range[1] - lon_range[0]) * 1.4) + 2)
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")
    scal = fields.sample(lon2d, lat2d, level, t)
    return grid_mesh(lon2d, lat2d, np.full_like(lon2d, r), scal)


def banded_face(fields, t, lat, lon, n_per_layer=6):
    """Flat cut face banded by the layer fields.

    Exactly one of lat/lon is a scalar (the fixed coordinate of the face).
    """
    fixed_lat = np.isscalar(lat)
    sweep = np.asarray(lon if fixed_lat else lat)
    parts, s_parts = [], []
    for layer in reversed(range(N_LAYERS)):
        r_out, r_in = layer_radii(layer)
        r = np.linspace(r_in, r_out, n_per_layer)
        r2d, sweep2d = np.meshgrid(r, sweep, indexing="ij")
        lat2d = np.full_like(r2d, lat) if fixed_lat else sweep2d
        lon2d = sweep2d if fixed_lat else np.full_like(r2d, lon)
        parts.append((r2d, lat2d, lon2d))
        s_parts.append(fields.sample(lon2d, lat2d, layer, t))
    r2d = np.concatenate([p[0] for p in parts], axis=0)
    lat2d = np.concatenate([p[1] for p in parts], axis=0)
    lon2d = np.concatenate([p[2] for p in parts], axis=0)
    scal = np.concatenate(s_parts, axis=0)
    return grid_mesh(lon2d, lat2d, r2d, scal)


def layer_separators():
    """Thin dark lines marking layer boundaries on the cut faces."""
    lat_arc = np.linspace(0.4, 90, 90)
    lon_arc = np.linspace(0, WEDGE, 90)
    lines = []
    for layer in range(N_LAYERS):
        _, r_in = layer_radii(layer)
        # vertical arcs on the two meridional walls
        lines.append(lonlat_to_xyz(np.full_like(lat_arc, -0.25), lat_arc, r_in))
        lines.append(
            lonlat_to_xyz(np.full_like(lat_arc, WEDGE + 0.25), lat_arc, r_in)
        )
        # concentric quarter-rings on the equatorial shelf
        lines.append(lonlat_to_xyz(lon_arc, np.full_like(lon_arc, 0.3), r_in))
    out = pv.lines_from_points(lines[0])
    for pts in lines[1:]:
        out += pv.lines_from_points(pts)
    return out


def healpix_grid_lines(nside_grid=8, radius=1.004):
    """Green HEALPix pixel boundaries on the outer sphere (outside wedge)."""
    lines = []
    for pix in range(healpy.nside2npix(nside_grid)):
        b = healpy.boundaries(nside_grid, pix, step=6, nest=True)  # 3 x N
        lon, lat = healpy.vec2ang(b.T, lonlat=True)
        lon = lon % 360.0
        if np.any((lon > 1.0) & (lon < WEDGE - 1.0) & (lat > 1.0)):
            continue
        pts = lonlat_to_xyz(lon, lat, radius)
        pts = np.vstack([pts, pts[:1]])
        lines.append(pv.lines_from_points(pts))
    out = lines[0]
    for ln in lines[1:]:
        out += ln
    return out


def render_globe(fields, t, path, size=1100):
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("white")
    common = dict(
        cmap=CMAP,
        clim=CLIM,
        show_scalar_bar=False,
        lighting=False,
    )
    # outer sphere minus the octant: full southern half + partial northern half
    pl.add_mesh(sphere_patch(fields, t, 0, R_OUT, (-90, 0), (0, 360)), **common)
    pl.add_mesh(sphere_patch(fields, t, 0, R_OUT, (0, 90), (WEDGE, 360)), **common)
    # cut faces: two meridional quarter-disk walls + equatorial shelf
    pl.add_mesh(banded_face(fields, t, np.linspace(0, 90, 130), 0.0), **common)
    pl.add_mesh(banded_face(fields, t, np.linspace(0, 90, 130), WEDGE), **common)
    pl.add_mesh(banded_face(fields, t, 0.0, np.linspace(0, WEDGE, 130)), **common)
    # floor of the cutout: the deepest level's surface
    pl.add_mesh(
        sphere_patch(fields, t, N_LEVELS, R_CORE, (0, 90), (0, WEDGE)), **common
    )
    pl.add_mesh(
        layer_separators(),
        color="#222233",
        line_width=1.2,
        opacity=0.55,
        lighting=False,
    )
    pl.add_mesh(
        healpix_grid_lines(),
        color="#46d97a",
        line_width=2,
        opacity=0.75,
    )
    pl.add_mesh(
        pv.Sphere(radius=1.04, theta_resolution=120, phi_resolution=120),
        color="#b8c0e8",
        opacity=0.13,
        specular=0.4,
        smooth_shading=True,
    )
    # camera looking into the wedge, north up, slight downward tilt
    cam_lonlat = lonlat_to_xyz(35.0, 22.0, 5.2)
    pl.camera.position = tuple(cam_lonlat)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)
    pl.screenshot(path, transparent_background=True)
    pl.close()


def pear_image(px=320):
    font = None
    for fsize in (160, 137, 96, 64):
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Apple Color Emoji.ttc", fsize
            )
            break
        except OSError:
            continue
    img = Image.new("RGBA", (font.size + 40, font.size + 40), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((20, 20), "\U0001f350", font=font, embedded_color=True)
    bbox = img.getbbox()
    img = img.crop(bbox).resize((px, px), Image.LANCZOS)
    return np.asarray(img)


def crop_alpha(img, pad=8):
    mask = img[..., 3] > 0.01
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    return img[r0 : r1 + 1 + pad, c0 : c1 + 1 + pad]


def compose(globe0_path, globe1_path, out_base):
    g0 = crop_alpha(plt.imread(globe0_path))
    g1 = crop_alpha(plt.imread(globe1_path))
    pear = pear_image()

    fig = plt.figure(figsize=(12, 5.2))
    label_kw = dict(fontsize=46, color="#555555", ha="center")

    ax0 = fig.add_axes([0.015, 0.02, 0.36, 0.78])
    ax0.imshow(g0)
    ax0.axis("off")
    fig.text(0.195, 0.84, r"$t_0$", **label_kw)

    axp = fig.add_axes([0.42, 0.42, 0.16, 0.42])
    axp.imshow(pear)
    axp.axis("off")

    axa = fig.add_axes([0.375, 0.05, 0.25, 0.40])
    axa.axis("off")
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    axa.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=4.5, color="black", mutation_scale=45),
    )

    ax1 = fig.add_axes([0.625, 0.02, 0.36, 0.78])
    ax1.imshow(g1)
    ax1.axis("off")
    fig.text(0.805, 0.84, r"$t_0 + \Delta t$", **label_kw)

    fig.patches.append(
        plt.Rectangle(
            (0.004, 0.008),
            0.992,
            0.984,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#1a1a1a",
            linewidth=6,
        )
    )
    fig.savefig(f"{out_base}.png", dpi=200, facecolor="white")
    fig.savefig(f"{out_base}.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fields = Fields()
    tmp = HERE / ".figure_timestep_tmp"
    tmp.mkdir(exist_ok=True)
    render_globe(fields, t=0, path=tmp / "globe_t0.png")
    render_globe(fields, t=1, path=tmp / "globe_t1.png")
    compose(tmp / "globe_t0.png", tmp / "globe_t1.png", HERE / "figure_timestep")
    print("wrote", HERE / "figure_timestep.png")
