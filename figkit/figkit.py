"""figkit: shared drawing kit for the TDSC-SI26 overview / result figures.

Vector-only: Phosphor duotone icons (MIT, icons/LICENSE-phosphor-MIT.txt) are parsed from SVG into matplotlib Paths,
so every figure is a pure-vector PDF with embedded TrueType (Type 42) Arial text.

Palette: the validated reference light palette of the dataviz method (categorical slots in fixed order, status colors
reserved for state).  Diagram canvases use points as data units (1 unit = 1/72 in), so icon sizes and font sizes share
one scale.
"""
import math
import pathlib
import re

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path

HERE = pathlib.Path(__file__).resolve().parent
ICON_DIR = HERE / "icons"

# ---------------------------------------------------------------- palette
C = dict(
    blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
    green="#008300", violet="#6250d6", red="#e34948",
    ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", plane="#f9f9f7",
    good="#0ca30c", warn="#fab219", serious="#ec835a", critical="#d03b3b", goodtext="#006300",
)


def tint(hex_color, t):
    """Mix a color with white: t=1 keeps the color, t=0 gives white."""
    r, g, b = mpl.colors.to_rgb(hex_color)
    return (1 - t + t * r, 1 - t + t * g, 1 - t + t * b)


def shade(hex_color, t):
    """Mix a color with black: t=0 keeps the color."""
    r, g, b = mpl.colors.to_rgb(hex_color)
    return (r * (1 - t), g * (1 - t), b * (1 - t))


# ---------------------------------------------------------------- fonts / rc
def setup():
    sup = pathlib.Path("/System/Library/Fonts/Supplemental")
    for f in ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf",
              "Arial Narrow.ttf", "Arial Narrow Bold.ttf", "Arial Narrow Italic.ttf"]:
        p = sup / f
        if p.exists():
            fm.fontManager.addfont(str(p))
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
        "axes.linewidth": 0.6, "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 600, "figure.dpi": 150,
    })


NARROW = "Arial Narrow"

# ---------------------------------------------------------------- SVG path parsing
_CMD = set("MmLlHhVvCcSsQqTtAaZz")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class _Scanner:
    def __init__(self, s):
        self.s, self.i = s, 0

    def skip(self):
        while self.i < len(self.s) and self.s[self.i] in " ,\t\n\r":
            self.i += 1

    def at_cmd(self):
        self.skip()
        return self.i < len(self.s) and self.s[self.i] in _CMD

    def done(self):
        self.skip()
        return self.i >= len(self.s)

    def cmd(self):
        self.skip()
        c = self.s[self.i]
        self.i += 1
        return c

    def num(self):
        self.skip()
        m = _NUM.match(self.s, self.i)
        if not m:
            raise ValueError(f"number expected at {self.i}: {self.s[self.i:self.i + 20]!r}")
        self.i = m.end()
        return float(m.group())

    def flag(self):
        self.skip()
        c = self.s[self.i]
        if c not in "01":
            raise ValueError("arc flag expected")
        self.i += 1
        return float(c)


def _arc_to_cubics(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    """SVG endpoint arc -> list of cubic bezier control triples (SVG spec F.6.5)."""
    if (x1, y1) == (x2, y2):
        return []
    if rx == 0 or ry == 0:
        return [((x1, y1), (x2, y2), (x2, y2))]
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1p ** 2) / rx ** 2 + (y1p ** 2) / ry ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    co = math.sqrt(max(0.0, num / den)) if den else 0.0
    if fa == fs:
        co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dt > 0:
        dt -= 2 * math.pi
    elif fs and dt < 0:
        dt += 2 * math.pi
    nseg = max(1, int(math.ceil(abs(dt) / (math.pi / 2) - 1e-9)))
    d = dt / nseg
    k = 4 / 3 * math.tan(d / 4)
    out = []

    def pt(t):
        x, y = rx * math.cos(t), ry * math.sin(t)
        return cp * x - sp * y + cx, sp * x + cp * y + cy

    def der(t):
        x, y = -rx * math.sin(t), ry * math.cos(t)
        return cp * x - sp * y, sp * x + cp * y

    for i in range(nseg):
        ta, tb = t1 + i * d, t1 + (i + 1) * d
        pa, pb = pt(ta), pt(tb)
        da, db = der(ta), der(tb)
        c1 = (pa[0] + k * da[0], pa[1] + k * da[1])
        c2 = (pb[0] - k * db[0], pb[1] - k * db[1])
        out.append((c1, c2, pb))
    return out


def parse_path(d):
    """SVG path data -> (vertices, codes) in SVG coordinates."""
    sc = _Scanner(d)
    V, K = [], []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    last_c2 = None   # last cubic control point (for S)
    last_q = None    # last quadratic control point (for T)
    cmd = None
    while not sc.done():
        if sc.at_cmd():
            cmd = sc.cmd()
        elif cmd is None:
            raise ValueError("path must start with a command")
        rel = cmd.islower()
        C_ = cmd.upper()
        ox, oy = cur if rel else (0.0, 0.0)
        if C_ == "Z":
            V.append(start); K.append(Path.CLOSEPOLY)
            cur = start
            last_c2 = last_q = None
            continue
        if C_ == "M":
            x, y = sc.num() + ox, sc.num() + oy
            V.append((x, y)); K.append(Path.MOVETO)
            cur = start = (x, y)
            cmd = "l" if rel else "L"   # implicit lineto
            last_c2 = last_q = None
            continue
        if C_ == "L":
            x, y = sc.num() + ox, sc.num() + oy
            V.append((x, y)); K.append(Path.LINETO); cur = (x, y); last_c2 = last_q = None
        elif C_ == "H":
            x = sc.num() + ox
            cur = (x, cur[1]); V.append(cur); K.append(Path.LINETO); last_c2 = last_q = None
        elif C_ == "V":
            y = sc.num() + oy
            cur = (cur[0], y); V.append(cur); K.append(Path.LINETO); last_c2 = last_q = None
        elif C_ == "C":
            c1 = (sc.num() + ox, sc.num() + oy)
            c2 = (sc.num() + ox, sc.num() + oy)
            p = (sc.num() + ox, sc.num() + oy)
            V += [c1, c2, p]; K += [Path.CURVE4] * 3; cur = p; last_c2 = c2; last_q = None
        elif C_ == "S":
            c1 = (2 * cur[0] - last_c2[0], 2 * cur[1] - last_c2[1]) if last_c2 else cur
            c2 = (sc.num() + ox, sc.num() + oy)
            p = (sc.num() + ox, sc.num() + oy)
            V += [c1, c2, p]; K += [Path.CURVE4] * 3; cur = p; last_c2 = c2; last_q = None
        elif C_ == "Q":
            q = (sc.num() + ox, sc.num() + oy)
            p = (sc.num() + ox, sc.num() + oy)
            V += [q, p]; K += [Path.CURVE3] * 2; cur = p; last_q = q; last_c2 = None
        elif C_ == "T":
            q = (2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1]) if last_q else cur
            p = (sc.num() + ox, sc.num() + oy)
            V += [q, p]; K += [Path.CURVE3] * 2; cur = p; last_q = q; last_c2 = None
        elif C_ == "A":
            rx, ry, phi = sc.num(), sc.num(), sc.num()
            fa, fs = sc.flag(), sc.flag()
            x, y = sc.num() + ox, sc.num() + oy
            for c1, c2, p in _arc_to_cubics(cur[0], cur[1], rx, ry, phi, fa, fs, x, y):
                V += [c1, c2, p]; K += [Path.CURVE4] * 3
            cur = (x, y); last_c2 = last_q = None
        else:
            raise ValueError(f"unsupported command {cmd}")
    return V, K


_ICON_CACHE = {}


def load_icon(name):
    """Return [(Path, opacity), ...] in a unit box centred on 0 with y up."""
    if name in _ICON_CACHE:
        return _ICON_CACHE[name]
    txt = (ICON_DIR / f"{name}.svg").read_text()
    vb = re.search(r'viewBox="([^"]+)"', txt)
    x0, y0, w, h = (float(v) for v in vb.group(1).split()) if vb else (0, 0, 256, 256)
    layers = []
    for m in re.finditer(r"<path([^>]*)/?>", txt):
        attrs = m.group(1)
        d = re.search(r'\bd="([^"]+)"', attrs).group(1)
        op = re.search(r'opacity="([^"]+)"', attrs)
        V, K = parse_path(d)
        V = np.array(V, dtype=float)
        V[:, 0] = (V[:, 0] - x0) / w - 0.5
        V[:, 1] = 0.5 - (V[:, 1] - y0) / h
        layers.append((Path(V, K), float(op.group(1)) if op else 1.0))
    _ICON_CACHE[name] = layers
    return layers


def _icon_patches(name, transform, color, light=None, light_t=0.28, zorder=6, lw=0):
    pats = []
    for path, op in load_icon(name):
        if op < 1:
            fc = light if light is not None else tint(color, light_t)
        else:
            fc = color
        pats.append(PathPatch(path, transform=transform, facecolor=fc, edgecolor="none", lw=lw, zorder=zorder))
    return pats


def icon(ax, name, x, y, size, color, light=None, light_t=0.28, zorder=6, rot=0, mode="data"):
    """Draw an icon centred at (x, y).

    mode='data': (x, y) and size in data units of an equal-aspect canvas.
    mode='pt'  : (x, y) in data coordinates of any axes (log or linear), size in points.
    mode='axes': (x, y) in axes fraction, size in points.
    """
    if mode == "data":
        tr = mtransforms.Affine2D().scale(size).rotate_deg(rot).translate(x, y) + ax.transData
    else:
        base = ax.transData if mode == "pt" else ax.transAxes
        tr = (mtransforms.Affine2D().scale(size / 72.0).rotate_deg(rot) + ax.figure.dpi_scale_trans
              + mtransforms.ScaledTranslation(x, y, base))
    for p in _icon_patches(name, tr, color, light, light_t, zorder):
        p.set_clip_on(False)
        ax.add_patch(p)


# ---------------------------------------------------------------- diagram primitives
def canvas(w_in, h_in):
    """Figure whose single axes uses points as data units (equal aspect, no frame)."""
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in * 72)
    ax.set_ylim(0, h_in * 72)
    ax.set_axis_off()
    return fig, ax


def rbox(ax, x, y, w, h, fc="white", ec=None, lw=0.6, r=4, zorder=2, ls="-", alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                       ec=ec if ec is not None else "none", lw=lw, zorder=zorder, ls=ls, alpha=alpha)
    p.set_clip_on(False)
    ax.add_patch(p)
    return p


def arrow(ax, p1, p2, color=None, lw=0.9, rad=0.0, head=5, zorder=4, ls="-", shrinkA=0, shrinkB=0, style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=f"{style},head_length={head * 0.11},head_width={head * 0.07}",
                        connectionstyle=f"arc3,rad={rad}", color=color or C["ink2"], lw=lw, zorder=zorder,
                        shrinkA=shrinkA, shrinkB=shrinkB, mutation_scale=10, linestyle=ls, capstyle="round",
                        joinstyle="round")
    a.set_clip_on(False)
    ax.add_patch(a)
    return a


def badge(ax, x, y, label, color, r=5.2, fs=6.5, zorder=8):
    ax.add_patch(Circle((x, y), r, fc=color, ec="white", lw=0.8, zorder=zorder, clip_on=False))
    ax.text(x, y - 0.15, label, ha="center", va="center", color="white", fontsize=fs, fontweight="bold",
            zorder=zorder + 1)


def text(ax, x, y, s, fs=6.5, color=None, ha="center", va="center", weight="normal", style="normal",
         family=None, zorder=9, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color or C["ink"], ha=ha, va=va, fontweight=weight,
                   fontstyle=style, family=family or mpl.rcParams["font.family"], zorder=zorder, **kw)


def chip(ax, x, y, s, fc, tc="white", fs=5.8, padx=3.0, h=8.5, zorder=8, family=None, weight="bold", ec="none",
         lw=0.5, ha="center"):
    """Pill-shaped label centred at (x, y) (or left-anchored with ha='left'); width measured from the text."""
    t = ax.text(0, 0, s, fontsize=fs, family=family or mpl.rcParams["font.family"], fontweight=weight)
    fig = ax.figure
    fig.canvas.draw() if not hasattr(fig.canvas, "get_renderer") else None
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    t.remove()
    wpt = bb.width * 72.0 / fig.dpi
    w = wpt + 2 * padx
    x0 = x - w / 2 if ha == "center" else x
    rbox(ax, x0, y - h / 2, w, h, fc=fc, ec=ec, lw=lw, r=h / 2, zorder=zorder)
    ax.text(x0 + w / 2, y - 0.2, s, fontsize=fs, color=tc, ha="center", va="center", zorder=zorder + 1,
            family=family or mpl.rcParams["font.family"], fontweight=weight)
    return w


def text_width(ax, s, fs, family=None, weight="normal"):
    t = ax.text(0, 0, s, fontsize=fs, family=family or mpl.rcParams["font.family"], fontweight=weight)
    r = ax.figure.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    t.remove()
    return bb.width * 72.0 / ax.figure.dpi


def hexagon(cx, cy, R, rot=0.0):
    a = np.radians(rot + np.arange(6) * 60 + 30)
    return np.c_[cx + R * np.cos(a), cy + R * np.sin(a)]


def panel(ax, x, y, w, h, color, title, num=None, title_fs=7.2, head_h=13, fill_t=0.07, r=5, icon_name=None):
    """Panel with a tinted header strip, a step badge and a title."""
    rbox(ax, x, y, w, h, fc=tint(color, fill_t), ec=tint(color, 0.35), lw=0.6, r=r, zorder=1)
    rbox(ax, x, y + h - head_h, w, head_h, fc=tint(color, 0.18), ec="none", r=r, zorder=1.2)
    # square off the lower corners of the header strip
    ax.add_patch(Polygon([[x, y + h - head_h], [x + w, y + h - head_h], [x + w, y + h - head_h + r],
                          [x, y + h - head_h + r]], closed=True, fc=tint(color, 0.18), ec="none", zorder=1.25))
    tx = x + 6
    if num is not None:
        badge(ax, x + 9, y + h - head_h / 2, str(num), color)
        tx = x + 17
    text(ax, tx, y + h - head_h / 2, title, fs=title_fs, ha="left", weight="bold", color=shade(color, 0.35))
    if icon_name:
        icon(ax, icon_name, x + w - 9, y + h - head_h / 2, 10, shade(color, 0.15))


def save(fig, path, png=True):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches=None, pad_inches=0, metadata={"Creator": "figkit", "Producer": None})
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
