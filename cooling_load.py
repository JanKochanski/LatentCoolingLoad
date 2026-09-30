#!/usr/bin/env python3
"""
Cooling capacity and required coil surface temperature for an air stream
(with dehumidification), based on CoolProp humid-air properties.

Method
------
1. Humid-air states (humidity ratio w, enthalpy h, specific volume v) are
   evaluated with CoolProp's HAPropsSI [1, 2].
2. Cooling capacity = dry-air mass flow * (h_in - h_out)  (steady-flow energy
   balance). The process is split into a latent part (dehumidification at
   inlet temperature) and a sensible part (cooling at constant outlet
   humidity ratio). Condensate enthalpy is reported separately.
3. Coil surface temperature via the "apparatus dew point" (ADP) /
   bypass-factor model [5]: the outlet air is treated as an adiabatic mixture
   of a "bypass" fraction of inlet air (BF) and a "contact" fraction (1-BF)
   that leaves saturated at the coil surface temperature. Mixing of moist
   air is exactly linear in the h-w plane (follows directly from the dry-air,
   water and energy balances), so the ADP is the point where the straight
   line inlet -> outlet, extended beyond the outlet, hits the saturation
   curve. Its temperature is the effective (mean) coil surface temperature.

Sources (freely accessible)
---------------------------
[1] Bell, I. H., Wronski, J., Quoilin, S., Lemort, V. (2014): Pure and
    Pseudo-Pure Fluid Thermophysical Property Evaluation and the Open-Source
    Thermophysical Property Library CoolProp. Ind. Eng. Chem. Res. 53(6),
    2498-2508. DOI: 10.1021/ie4033999 (author manuscript on PubMed Central)
[2] CoolProp documentation, "Humid Air Properties" (input keys such as
    Hda, Vda, W, R; formulation and references):
    http://www.coolprop.org/fluid_properties/HumidAir.html
[3] Wagner, W., Pruss, A. (2002): The IAPWS Formulation 1995 for the
    Thermodynamic Properties of Ordinary Water Substance for General and
    Scientific Use. J. Phys. Chem. Ref. Data 31, 387-535.
    DOI: 10.1063/1.1461829 (release also available at https://iapws.org)
[4] Lemmon, E. W., Jacobsen, R. T., Penoncello, S. G., Friend, D. G. (2000):
    Thermodynamic Properties of Air and Mixtures of Nitrogen, Argon, and
    Oxygen From 60 to 2000 K at Pressures to 2000 MPa. J. Phys. Chem. Ref.
    Data 29, 331-385. DOI: 10.1063/1.1285884
[5] U.S. DOE: EnergyPlus Engineering Reference (open documentation),
    chapters on cooling coil models using apparatus dew point and bypass
    factor. https://energyplus.net/documentation

Limitations
-----------
* Real coils have a varying surface temperature; the ADP is an effective
  value. The coolant (chilled water / evaporating refrigerant) must be
  colder than this value to drive the heat transfer.
* Ideal, condensation-only process on a coil; no frost model.

Examples
--------
    python cooling_load.py --flow 2000 --unit m3h --t-in 32 --rh-in 60 --t-out 14 --rh-out 90
    python cooling_load.py --flow 500 --unit ls --t-in 28 --rh-in 55 --t-out 16 --rh-out 80 --p 100500
"""

import argparse
from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI

P_STD = 101325.0  # Pa


@dataclass
class CoolingResult:
    m_dry_air: float      # kg/s dry air
    V_in: float           # m3/s at inlet state
    V_out: float          # m3/s at outlet state
    w_in: float           # kg water / kg dry air
    w_out: float
    h_in: float           # kJ/kg dry air
    h_out: float
    dew_in: float         # degC
    dew_out: float
    q_total: float        # kW (air side)
    q_sensible: float     # kW
    q_latent: float       # kW
    condensate: float     # kg/s
    q_condensate: float   # kW, enthalpy carried away by condensate
    q_net: float          # kW, air side minus condensate enthalpy
    shr: float            # sensible heat ratio
    # Coil surface (ADP) model; None if not determinable
    t_surface: float | None = None       # degC, effective surface temperature (ADP)
    bypass_factor: float | None = None   # BF
    contact_factor: float | None = None  # 1 - BF
    w_adp: float | None = None           # kg/kg, humidity ratio at ADP
    surface_note: str = ""


def _to_m3s(flow: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ("m3h", "m3/h"):
        return flow / 3600.0
    if unit in ("ls", "l/s"):
        return flow / 1000.0
    if unit in ("m3s", "m3/s"):
        return flow
    raise ValueError(f"Unknown flow unit: {unit}")


def _find_adp_temperature(h_in: float, w_in: float, h_out: float, w_out: float,
                          T_out: float, p: float):
    """Locate the apparatus dew point (ADP) on the line inlet -> outlet.

    Looks for a saturated state (h_s, w_s) that is collinear with inlet and
    outlet in the (w, h) plane and lies beyond the outlet, i.e. for which the
    outlet is a mixture of inlet air and that saturated air. Only valid
    (non-supersaturated) states are evaluated: the saturation temperature T_s
    is varied, starting at the outlet temperature and going down, and the
    first sign change of the collinearity residual is refined by bisection.
    Enthalpies in J/kg dry air, humidity ratios in kg/kg, temperatures in K.
    Returns T_s in K or None if the line never meets the saturation curve.
    """

    def saturated(T_s: float):
        w_s = HAPropsSI("W", "T", T_s, "R", 1.0, "P", p)
        h_s = HAPropsSI("Hda", "T", T_s, "W", w_s, "P", p)
        return h_s, w_s

    def residual(T_s: float):
        """Cross product (zero when collinear) and line parameter t (out at t = 1)."""
        h_s, w_s = saturated(T_s)
        f = (h_s - h_in) * (w_out - w_in) - (w_s - w_in) * (h_out - h_in)
        return f, (w_s - w_in) / (w_out - w_in)

    if w_out >= saturated(T_out)[1] * (1.0 - 1e-9):  # outlet already saturated
        return T_out

    step, T_min = 0.1, 223.15
    T_prev = T_out
    f_prev, _ = residual(T_prev)
    T = T_out - step
    while T >= T_min:
        try:
            f, t_par = residual(T)
        except ValueError:  # CoolProp cannot evaluate this state
            return None
        if f_prev * f <= 0.0 and t_par >= 1.0 - 1e-6:
            lo, hi = T, T_prev  # bisection between the bracketing temperatures
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if residual(lo)[0] * residual(mid)[0] <= 0.0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        T_prev, f_prev = T, f
        T -= step
    return None


def cooling_load(
    flow: float,
    unit: str,
    t_in_c: float,
    rh_in_pct: float,
    t_out_c: float,
    rh_out_pct: float,
    p_pa: float = P_STD,
    flow_ref: str = "in",
    t_condensate_c: float | None = None,
) -> CoolingResult:
    """Compute the required cooling capacity and coil surface temperature.

    flow_ref: state at which the given volume flow is specified ("in" or "out").
    t_condensate_c: condensate temperature (default: outlet temperature).
    """
    T_in, T_out = t_in_c + 273.15, t_out_c + 273.15
    rh_in, rh_out = rh_in_pct / 100.0, rh_out_pct / 100.0
    T_cond = (t_condensate_c if t_condensate_c is not None else t_out_c) + 273.15

    # Humidity ratios [kg/kg dry air]
    w_in = HAPropsSI("W", "T", T_in, "R", rh_in, "P", p_pa)
    w_out = HAPropsSI("W", "T", T_out, "R", rh_out, "P", p_pa)

    # Specific volumes [m3/kg dry air] and dry-air mass flow
    v_in = HAPropsSI("Vda", "T", T_in, "W", w_in, "P", p_pa)
    v_out = HAPropsSI("Vda", "T", T_out, "W", w_out, "P", p_pa)
    V = _to_m3s(flow, unit)
    m_da = V / (v_in if flow_ref == "in" else v_out)

    # Enthalpies [J/kg dry air]
    h_in = HAPropsSI("Hda", "T", T_in, "W", w_in, "P", p_pa)
    h_mid = HAPropsSI("Hda", "T", T_in, "W", w_out, "P", p_pa)  # dehumidified at T_in
    h_out = HAPropsSI("Hda", "T", T_out, "W", w_out, "P", p_pa)

    q_total = m_da * (h_in - h_out)
    q_lat = m_da * (h_in - h_mid)
    q_sens = m_da * (h_mid - h_out)

    # Condensate and the enthalpy it carries away (saturated liquid water, [3]).
    # The humid-air enthalpy convention counts liquid water at 0 degC as zero,
    # so the liquid enthalpy is taken relative to the triple point.
    m_cond = m_da * (w_in - w_out)
    h_liq = PropsSI("H", "T", T_cond, "Q", 0, "Water")  # J/kg
    h_liq_ref = PropsSI("H", "T", 273.16, "Q", 0, "Water")
    q_cond = m_cond * (h_liq - h_liq_ref)
    q_net = q_total - q_cond

    dew_in = HAPropsSI("D", "T", T_in, "W", w_in, "P", p_pa) - 273.15
    dew_out = HAPropsSI("D", "T", T_out, "W", w_out, "P", p_pa) - 273.15

    # ---- Coil surface temperature (ADP / bypass-factor model, [5]) ----------
    t_surface = bypass = contact = w_adp = None
    note = ""
    if w_out > w_in + 1e-6:
        note = "Not applicable: outlet is more humid than inlet (humidification)."
    elif w_in - w_out < 1e-6:
        note = (
            "Dry coil (no dehumidification): the surface temperature is not fixed "
            f"by the air states. It must stay above the inlet dew point ({dew_in:.1f} degC) "
            f"to remain dry and below the outlet temperature ({t_out_c:.1f} degC); "
            "the exact value depends on the coil's bypass factor."
        )
    else:
        T_adp = _find_adp_temperature(h_in, w_in, h_out, w_out, T_out, p_pa)
        if T_adp is None:
            note = (
                "No coil surface temperature found: the line inlet -> outlet never reaches "
                "saturation, so this outlet state cannot be produced by a cooling coil alone "
                "(air that dry at this temperature needs cooling below the outlet temperature "
                "plus reheat; air that humid needs a lower outlet RH or colder outlet)."
            )
        else:
            w_adp = HAPropsSI("W", "T", T_adp, "R", 1.0, "P", p_pa)
            t_surface = T_adp - 273.15
            t_line = (w_adp - w_in) / (w_out - w_in)  # line parameter: in = 0, out = 1, ADP = t_line >= 1
            contact = 1.0 / t_line                    # fraction of air that reaches the surface state
            bypass = 1.0 - contact                    # = (t_line - 1) / t_line
            if t_surface < 0.0:
                note = "Surface temperature below 0 degC: frost formation on the coil is likely."

    return CoolingResult(
        m_dry_air=m_da,
        V_in=m_da * v_in,
        V_out=m_da * v_out,
        w_in=w_in,
        w_out=w_out,
        h_in=h_in / 1e3,
        h_out=h_out / 1e3,
        dew_in=dew_in,
        dew_out=dew_out,
        q_total=q_total / 1e3,
        q_sensible=q_sens / 1e3,
        q_latent=q_lat / 1e3,
        condensate=m_cond,
        q_condensate=q_cond / 1e3,
        q_net=q_net / 1e3,
        shr=q_sens / q_total if q_total else float("nan"),
        t_surface=t_surface,
        bypass_factor=bypass,
        contact_factor=contact,
        w_adp=w_adp,
        surface_note=note,
    )


def format_report(r: CoolingResult) -> str:
    """Return a plain-text report (used by the CLI and the GUI)."""
    lines = [
        "--- Air states ---",
        f"Inlet : x = {r.w_in * 1000:6.2f} g/kg, h = {r.h_in:6.2f} kJ/kg, dew point = {r.dew_in:5.1f} degC",
        f"Outlet: x = {r.w_out * 1000:6.2f} g/kg, h = {r.h_out:6.2f} kJ/kg, dew point = {r.dew_out:5.1f} degC",
        "--- Flows ---",
        f"Dry air mass flow : {r.m_dry_air:.4f} kg/s ({r.m_dry_air * 3600:.1f} kg/h)",
        f"Volume flow inlet : {r.V_in * 3600:.1f} m3/h ({r.V_in * 1000:.1f} l/s)",
        f"Volume flow outlet: {r.V_out * 3600:.1f} m3/h ({r.V_out * 1000:.1f} l/s)",
        "--- Cooling capacity ---",
        f"Total (air side)  : {r.q_total:8.2f} kW",
        f"  Sensible        : {r.q_sensible:8.2f} kW",
        f"  Latent          : {r.q_latent:8.2f} kW",
        f"  SHR             : {r.shr:8.2f}",
        f"Condensate        : {r.condensate * 3600:8.2f} kg/h",
        f"Net (minus condensate enthalpy {r.q_condensate:.3f} kW): {r.q_net:.2f} kW",
        "--- Coil surface (ADP model) ---",
    ]
    if r.t_surface is not None:
        lines += [
            f"Eff. surface temp. (ADP): {r.t_surface:6.2f} degC",
            f"Bypass factor BF        : {r.bypass_factor:6.3f}",
            f"Contact factor 1-BF     : {r.contact_factor:6.3f}",
        ]
    if r.surface_note:
        lines.append(r.surface_note)

    if r.w_out > r.w_in + 1e-6:
        lines.append("\nWARNING: outlet humidity ratio is higher than inlet -> this needs "
                     "humidification, not just cooling. Check your outlet values.")
    elif r.q_total < 0:
        lines.append("\nWARNING: negative capacity -> this is a heating process.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cooling capacity for an air stream (CoolProp).")
    ap.add_argument("--flow", type=float, required=True, help="Volume flow")
    ap.add_argument("--unit", default="m3h", choices=["m3h", "ls", "m3s"], help="Flow unit (default: m3h)")
    ap.add_argument("--flow-ref", default="in", choices=["in", "out"],
                    help="State at which the volume flow is specified (default: in)")
    ap.add_argument("--t-in", type=float, required=True, help="Inlet temperature [degC]")
    ap.add_argument("--rh-in", type=float, required=True, help="Inlet relative humidity [%%]")
    ap.add_argument("--t-out", type=float, required=True, help="Outlet temperature [degC]")
    ap.add_argument("--rh-out", type=float, required=True, help="Outlet relative humidity [%%]")
    ap.add_argument("--p", type=float, default=P_STD, help="Pressure [Pa] (default: 101325)")
    ap.add_argument("--t-cond", type=float, default=None, help="Condensate temperature [degC] (default: outlet T)")
    args = ap.parse_args()

    result = cooling_load(
        args.flow, args.unit, args.t_in, args.rh_in, args.t_out, args.rh_out,
        p_pa=args.p, flow_ref=args.flow_ref, t_condensate_c=args.t_cond,
    )
    print(format_report(result))


if __name__ == "__main__":
    main()
