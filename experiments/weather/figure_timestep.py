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
import matplotlib.patheffects as patheffects
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

HERE = Path(__file__).parent

NSIDE = 64
N_LEVELS = 13
SAMPLE_PATH = HERE / "example_data" / "figure_sample_2019-06-01.npz"

# surface vars: [msl, u10, v10, t2m]; upper vars: [z, q, t, u, v],
# levels ordered 1000 hPa (index 0) -> 50 hPa
VARIABLES = {
    "t2m": dict(surface_var=3, upper_var=2),  # temperature
    "u10": dict(surface_var=1, upper_var=3),  # u-wind
}

# --- octant cutout / shell geometry --------------------------------------
# The solid inner sphere is the surface; the 13 upper levels stack outward
# (1000 hPa just above the surface, 50 hPa outermost).
WEDGE = 90.0  # octant: lon in [0, WEDGE], lat in [0, 90] removed
N_LAYERS = N_LEVELS  # shells: one per upper level
R_OUT = 1.0
DR_LEVEL = 0.034
CMAP = "plasma"


class Fields:
    """Real ERA5 validation sample on the HEALPix grid (nested ordering).

    Surface variable on the solid inner sphere, the upper variable on the
    13 pressure-level shells stacked above it.
    Produced by extract_sample_pair.py on the cluster.
    """

    def __init__(
        self,
        surface_var,
        upper_var,
        path=SAMPLE_PATH,
        denormalize=True,
        azimuth=0.0,
        upper_sigma=None,
        upper_trend=1.0,
    ):
        """upper_sigma / upper_trend: visualization middle ground between
        normalized and absolute. Each upper level's spatial variation is
        rendered with fixed amplitude `upper_sigma` (physical units), and the
        vertical trend of the level means is kept but compressed by the
        factor `upper_trend` (1 = true means, 0 = flat like normalized),
        anchored at the lowest level (1000 hPa)."""
        d = np.load(path)
        self.azimuth = azimuth  # rotates the data under the fixed cutout

        def field(key, var, mean, std):
            arr = d[key] * std + mean if denormalize else d[key]
            return arr[var].astype(np.float32)

        self.surface = {
            t: field(f"{key}_surface", surface_var, d["mean_surface"], d["std_surface"])
            for t, key in ((0, "input"), (1, "target"))
        }
        upper_std = upper_sigma if upper_sigma is not None else d["std_upper"]
        mean_upper = d["mean_upper"]
        if upper_trend != 1.0:
            anchor = mean_upper[:, :1]
            mean_upper = anchor + upper_trend * (mean_upper - anchor)
        self.upper = {
            t: field(f"{key}_upper", upper_var, mean_upper, upper_std)
            for t, key in ((0, "input"), (1, "target"))
        }
        self.time = str(d["time"])
        allv = np.concatenate([self.surface[0], self.upper[0].ravel()])
        self.clim = (np.percentile(allv, 1), np.percentile(allv, 99))

    def sample(self, lon, lat, level, t):
        """level: 0 = surface, 1..N_LEVELS = upper levels. t: 0 or 1."""
        pix = healpy.ang2pix(
            NSIDE, (np.asarray(lon) + self.azimuth) % 360.0, lat, nest=True, lonlat=True
        )
        if level == 0:
            return self.surface[t][pix]
        return self.upper[t][level - 1][pix]


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
    """Outer/inner radius of shell `layer` (0 = outermost = 50 hPa)."""
    r_out = R_OUT - layer * DR_LEVEL
    return r_out, r_out - DR_LEVEL


def shell_level(layer):
    """Field level of shell `layer`: outermost shell is the top of the
    atmosphere (50 hPa), the innermost is 1000 hPa."""
    return N_LEVELS - layer


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
        s_parts.append(fields.sample(lon2d, lat2d, shell_level(layer), t))
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
        clim=fields.clim,
        show_scalar_bar=False,
        lighting=False,
    )
    # outer sphere (50 hPa) minus the octant: full southern half + partial north
    top = shell_level(0)
    pl.add_mesh(sphere_patch(fields, t, top, R_OUT, (-90, 0), (0, 360)), **common)
    pl.add_mesh(sphere_patch(fields, t, top, R_OUT, (0, 90), (WEDGE, 360)), **common)
    # cut faces: two meridional quarter-disk walls + equatorial shelf
    pl.add_mesh(banded_face(fields, t, np.linspace(0, 90, 130), 0.0), **common)
    pl.add_mesh(banded_face(fields, t, np.linspace(0, 90, 130), WEDGE), **common)
    pl.add_mesh(banded_face(fields, t, 0.0, np.linspace(0, WEDGE, 130)), **common)
    # floor of the cutout: the solid inner sphere is the surface field
    pl.add_mesh(sphere_patch(fields, t, 0, R_CORE, (0, 90), (0, WEDGE)), **common)
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
    img = img.crop(img.getbbox())
    # resize with premultiplied alpha to avoid dark fringing at the edges
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    alpha = np.maximum(arr[..., 3:], 1e-3)
    arr[..., :3] = np.clip(arr[..., :3] / alpha, 0, 1)
    return arr


def crop_alpha(img, pad=8):
    mask = img[..., 3] > 0.01
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    return img[r0 : r1 + 1 + pad, c0 : c1 + 1 + pad]


def compose(globe0_path, globe1_path, out_base, pdf=True, transparent=True):
    g0 = crop_alpha(plt.imread(globe0_path))
    g1 = crop_alpha(plt.imread(globe1_path))
    pear = pear_image()

    fig = plt.figure(figsize=(12, 5.2))
    label_kw = dict(fontsize=46, color="white", ha="center")

    ax0 = fig.add_axes([0.015, 0.02, 0.36, 0.78])
    ax0.imshow(g0)
    ax0.axis("off")
    fig.text(0.195, 0.84, r"$t_0$", **label_kw)

    axp = fig.add_axes([0.42, 0.46, 0.16, 0.42])
    axp.imshow(pear)
    axp.axis("off")

    # arrow axis spans the same height as the globes so y=0.5 is the equator
    axa = fig.add_axes([0.375, 0.02, 0.25, 0.78])
    axa.axis("off")
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    ann = axa.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=4.5, color="white", mutation_scale=45),
    )
    ann.arrow_patch.set_path_effects(
        [patheffects.withStroke(linewidth=8, foreground="#2b2b2b")]
    )

    ax1 = fig.add_axes([0.625, 0.02, 0.36, 0.78])
    ax1.imshow(g1)
    ax1.axis("off")
    fig.text(0.805, 0.84, r"$t_0 + \Delta t$", **label_kw)

    save_kw = dict(transparent=True) if transparent else dict(facecolor="white")
    fig.savefig(f"{out_base}.png", dpi=200, **save_kw)
    if pdf:
        fig.savefig(f"{out_base}.pdf", **save_kw)
    plt.close(fig)


def contact_sheet(paths, labels, out_path, n_cols=2):
    n_rows = int(np.ceil(len(paths) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 3.4))
    for ax in np.asarray(axes).ravel():
        ax.axis("off")
    for ax, path, label in zip(np.asarray(axes).ravel(), paths, labels):
        ax.imshow(plt.imread(path))
        ax.set_title(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def final_figure(tmp):
    """The selected version: t2m, azimuth 0, sigma 12, trend 0.5."""
    fields = Fields(**VARIABLES["t2m"], azimuth=0, upper_sigma=12.0, upper_trend=0.5)
    globes = [tmp / f"globe_final_t{t}.png" for t in (0, 1)]
    for t, globe in enumerate(globes):
        if not globe.exists():
            render_globe(fields, t=t, path=globe)
    compose(*globes, HERE / "figure_timestep")
    print("wrote", HERE / "figure_timestep.png")


def variant_sweep(tmp):
    out_dir = HERE / "figure_timestep_variants"
    out_dir.mkdir(exist_ok=True)
    angles = range(0, 360, 45)
    for var_name, var in VARIABLES.items():
        for denormalize, suffix in ((True, ""), (False, "_normalized")):
            variants = []
            for az in angles:
                fields = Fields(**var, denormalize=denormalize, azimuth=az)
                globes = [
                    tmp / f"globe_{var_name}{suffix}_az{az:03d}_t{t}.png"
                    for t in (0, 1)
                ]
                for t, globe in enumerate(globes):
                    if not globe.exists():
                        render_globe(fields, t=t, path=globe)
                base = out_dir / f"{var_name}{suffix}_az{az:03d}"
                compose(*globes, base, pdf=False)
                variants.append(f"{base}.png")
                print("wrote", f"{base}.png", flush=True)
            sheet = out_dir / f"overview_{var_name}{suffix}.png"
            contact_sheet(variants, [f"azimuth {az}°" for az in angles], sheet)
            print("wrote", sheet, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="render all variable/normalization/azimuth variants",
    )
    args = parser.parse_args()
    tmp = HERE / ".figure_timestep_tmp"
    tmp.mkdir(exist_ok=True)
    if args.sweep:
        variant_sweep(tmp)
    else:
        final_figure(tmp)
