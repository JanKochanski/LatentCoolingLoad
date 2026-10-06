"""
Mollier (h-x) diagram for humid air with an air-conditioning process drawn in.

Coordinates
-----------
Classic Mollier layout with oblique axes: the abscissa is the humidity ratio
x [g/kg dry air] and the ordinate is Y = h - s * x, where s is the slope of
the 0 degC isotherm in the (x, h) plane (~2501 kJ/kg, i.e. the enthalpy of
vaporisation at 0 degC). This shear makes the 0 degC isotherm horizontal, the
lines of constant x vertical, and the lines of constant enthalpy diagonal.
In these coordinates a mixing process of two air streams is a straight line.

Sources (freely accessible)
---------------------------
[1] Bell, I. H., Wronski, J., Quoilin, S., Lemort, V. (2014): Pure and
    Pseudo-Pure Fluid Thermophysical Property Evaluation and the Open-Source
    Thermophysical Property Library CoolProp. Ind. Eng. Chem. Res. 53(6),
    2498-2508. DOI: 10.1021/ie4033999
[2] CoolProp documentation, "Humid Air Properties":
    http://www.coolprop.org/fluid_properties/HumidAir.html
    (saturation and enthalpy values; below 0 degC the "saturation" curve is
    the equilibrium over ice, as implemented in CoolProp)
"""

import math

import numpy as np
from CoolProp.HumidAirProp import HAPropsSI

C0 = 273.15  # K


def _shear(p: float) -> float:
    """Slope of the 0 degC isotherm in the (w [kg/kg], h [kJ/kg]) plane."""
    h0 = HAPropsSI("Hda", "T", C0, "W", 0.0, "P", p)
    h1 = HAPropsSI("Hda", "T", C0, "W", 0.02, "P", p)
    return (h1 - h0) / 0.02 / 1e3


def draw_mollier(fig, p_pa, state_in, state_out, state_adp=None, state_limit=None,
                 adp_label="ADP", state_room=None):
    """Draw a Mollier diagram with the process into a matplotlib Figure.

    state_*: tuples (temperature [degC], humidity ratio [kg/kg dry air]).
    state_adp: optional apparatus dew point (surface state), drawn as an
    extension of the process line (label via adp_label, e.g. "Surface" for a
    dry coil whose surface state is not saturated).
    state_room: optional room-air state; the humidity added by a moisture stream
    (same temperature, higher x) leads from there to state_in.
    state_limit: optional ideal-coil limit (BF = 0): the saturated state on the
    outlet isenthalp, drawn with a dotted isenthalp from the outlet.
    Returns the main Axes.
    """
    s = _shear(p_pa)

    def xy(t_c, w):
        h = HAPropsSI("Hda", "T", t_c + C0, "W", w, "P", p_pa) / 1e3
        return w * 1000.0, h - s * w

    def w_sat(t_c):
        return HAPropsSI("W", "T", t_c + C0, "R", 1.0, "P", p_pa)

    # ---- Plot ranges, derived from the process states ----------------------
    states = [state_in, state_out] + ([state_adp] if state_adp else []) + ([state_limit] if state_limit else []) + ([state_room] if state_room else [])
    t_lo = max(5 * math.floor((min(s_[0] for s_ in states) - 5) / 5), -20)
    t_hi = 5 * math.ceil((max(s_[0] for s_ in states) + 5) / 5)
    x_max_g = max(10.0, 2 * math.ceil(max(s_[1] for s_ in states) * 1000 * 1.3 / 2))
    x_max = x_max_g / 1000.0
    y_lo = xy(t_lo, 0.0)[1]
    y_hi = xy(t_hi, min(x_max, w_sat(t_hi)))[1]

    fig.clear()
    ax = fig.add_subplot(111)

    # ---- Lines of constant enthalpy (diagonal) ------------------------------
    h_span = y_hi - y_lo  # enthalpy range visible along the right border
    h_step = 5 if h_span <= 60 else (10 if h_span <= 150 else 20)
    h_first = math.ceil((y_lo + s * x_max) / h_step) * h_step
    h_vals = np.arange(h_first, y_hi + s * x_max + 1e-9, h_step)
    for h in h_vals:
        ax.plot([0, x_max_g], [h, h - s * x_max], color="#8fa9c9", lw=0.7, zorder=1)

    # ---- Isotherms (straight lines from x = 0 up to saturation) -------------
    t_ticks = list(range(int(t_lo), int(t_hi) + 1, 5))
    for t in t_ticks:
        x0, y0 = xy(t, 0.0)
        x1, y1 = xy(t, min(w_sat(t), x_max))
        ax.plot([x0, x1], [y0, y1], color="#e3c9c9", lw=0.6, zorder=1)

    # ---- Fog region (beyond saturation), masks the enthalpy lines -----------
    sat_t = np.linspace(t_lo, t_hi + 15, 90)
    sat_pts = [xy(t, w_sat(t)) for t in sat_t]
    fog_x = [p_[0] for p_ in sat_pts] + [5 * x_max_g, 5 * x_max_g, sat_pts[0][0]]
    fog_y = [p_[1] for p_ in sat_pts] + [sat_pts[-1][1], y_lo - 1000, y_lo - 1000]
    ax.fill(fog_x, fog_y, facecolor="#f2f2f2", edgecolor="none", zorder=2)

    # ---- Relative-humidity curves --------------------------------------------
    t_grid = np.linspace(t_lo, t_hi, 60)
    for phi in np.arange(0.1, 1.0, 0.1):
        pts = [xy(t, HAPropsSI("W", "T", t + C0, "R", float(phi), "P", p_pa)) for t in t_grid]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#8fb08f", lw=0.7, zorder=3)
        inside = [i for i, (x_, y_) in enumerate(pts)
                  if x_ <= 0.97 * x_max_g and y_lo <= y_ <= y_hi]
        if inside:
            i = inside[-1]
            ax.text(xs[i], ys[i], f"{phi * 100:.0f}%", fontsize=7, color="#5a805a",
                    ha="right", va="bottom", zorder=3)
    ax.plot([p_[0] for p_ in sat_pts], [p_[1] for p_ in sat_pts], color="#2f6f2f", lw=1.5, zorder=4)

    # ---- Process -------------------------------------------------------------
    p_in = xy(*state_in)
    p_out = xy(*state_out)
    if state_room is not None:
        p_room = xy(*state_room)
        ax.plot([p_room[0], p_in[0]], [p_room[1], p_in[1]], color="#777777", lw=1.4, ls="--", zorder=6)
        ax.scatter(*p_room, color="#777777", zorder=7)
        ax.annotate(f"Room ({state_room[0]:.1f} °C)", p_room, textcoords="offset points",
                    xytext=(-8, 6), fontsize=8, color="#555555", ha="right", zorder=7)
    ax.annotate("", xy=p_out, xytext=p_in, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="#c00000", lw=2))
    ax.scatter(*p_in, color="#c00000", zorder=7)
    ax.scatter(*p_out, color="#c00000", zorder=7)
    ax.annotate(f"In ({state_in[0]:.1f} °C)", p_in, textcoords="offset points",
                xytext=(6, 6), fontsize=8, color="#c00000", zorder=7)
    ax.annotate(f"Out ({state_out[0]:.1f} °C)", p_out, textcoords="offset points",
                xytext=(8, 3), fontsize=8, color="#c00000", zorder=7)
    if state_adp is not None:
        p_adp = xy(*state_adp)
        ax.plot([p_out[0], p_adp[0]], [p_out[1], p_adp[1]], color="#c00000",
                lw=1.2, ls="--", zorder=6)
        ax.scatter(*p_adp, color="#c00000", marker="s", zorder=7)
        ax.annotate(f"{adp_label} ({state_adp[0]:.1f} °C)", p_adp, textcoords="offset points",
                    xytext=(-8, 8), fontsize=8, color="#c00000", ha="right", zorder=7)
    if state_limit is not None:
        p_lim = xy(*state_limit)
        ax.plot([p_out[0], p_lim[0]], [p_out[1], p_lim[1]], color="#444444",
                lw=1.0, ls=":", zorder=6)
        ax.scatter(*p_lim, color="#444444", marker="D", s=22, zorder=7)
        ax.annotate(f"BF=0 limit ({state_limit[0]:.1f} °C)", p_lim, textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color="#444444", zorder=7)

    # ---- Axes ------------------------------------------------------------------
    ax.set_xlim(0, x_max_g)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("x  [g/kg dry air]")
    ax.set_yticks([xy(t, 0.0)[1] for t in t_ticks])
    ax.set_yticklabels([str(t) for t in t_ticks])
    ax.set_ylabel("t  [°C]  (isotherms)")

    ax_h = ax.twinx()  # enthalpy lines end at the right border
    ax_h.set_ylim(y_lo, y_hi)
    ax_h.set_yticks([h - s * x_max for h in h_vals])
    ax_h.set_yticklabels([f"{h:.0f}" for h in h_vals])
    ax_h.set_ylabel("h  [kJ/kg dry air]  (diagonal lines)")

    ax.set_title(f"Mollier h-x diagram (p = {p_pa / 100:.1f} hPa)", fontsize=10)
    fig.set_layout_engine("tight")  # re-applied on every draw, so it follows window resizing
    return ax
