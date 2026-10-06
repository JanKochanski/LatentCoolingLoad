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
4. Ideal-coil limit: the point on the saturation curve with the same enthalpy
   as the outlet (isenthalp x saturation curve) is the BF = 0 case, i.e. the
   warmest surface that could produce this capacity (upper bound of T_s; close
   to the wet-bulb temperature of the outlet state). Real coils (BF > 0) need
   a colder surface.
5. Second mode (bypass factor given): with BF and the target outlet enthalpy
   (or capacity) the mixing rule h_out = BF*h_in + (1-BF)*h_s fixes the
   surface-state enthalpy h_s. If the saturated state with h_s is drier than
   the inlet air the coil is wet: T_s is the saturation temperature at h_s
   and x_out = BF*x_in + (1-BF)*x_s. Otherwise the coil runs dry (x_out =
   x_in) and T_s follows from h_s at constant x_in. The outlet T and RH are
   then results, not inputs.
6. Capacity <-> volume flow: instead of the volume flow, the capacity (total,
   air side, i.e. m_dry_air * (h_in - h_out)) can be given and the volume flow
   is returned. In bypass-factor mode this needs both the capacity and the
   target outlet enthalpy, because the enthalpy difference links them.
7. Capacity type and moisture stream (capacity given, outlet T and RH given):
   the given capacity is either the total (sensible + latent) or only the
   sensible part, defined as m_dry_air * (h(T_in, x_out) - h(T_out, x_out)),
   i.e. cooling at the outlet humidity ratio. The latent part then follows
   from the dehumidification x_in -> x_out. With a sensible capacity an
   optional moisture stream (e.g. evaporation from a waste-heat source) can be
   added: the cooler inlet is the room air (T, RH) plus delta_x = m_water /
   m_dry_air at unchanged temperature. Because the sensible part does not
   depend on x_in, the dry-air flow follows directly (no iteration), and the
   moisture adds latent load m_dry_air * (h(T_in, x_room + delta_x) -
   h(T_in, x_room)). An evaporation power is converted to a mass flow with
   the enthalpy of vaporisation of water at the inlet temperature.

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
    # bypass factor given, target outlet enthalpy (or --q for capacity in kW):
    python cooling_load.py --flow 2000 --unit m3h --t-in 27 --rh-in 50 --bf 0.15 --h-out 36
    # capacity given, volume flow is the result (no --flow):
    python cooling_load.py --q 15 --t-in 27 --rh-in 50 --t-out 13 --rh-out 90
    python cooling_load.py --q 15 --t-in 27 --rh-in 50 --bf 0.15 --h-out 36
    # sensible capacity, optional moisture stream (--moisture-kgh or --moisture-kw):
    python cooling_load.py --q 10 --q-kind sensible --moisture-kgh 5 --t-in 27 --rh-in 50 --t-out 13 --rh-out 90
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
    # Ideal-coil limit (BF = 0): saturated state on the outlet isenthalp
    t_limit: float | None = None         # degC, upper bound of the surface temperature
    w_limit: float | None = None         # kg/kg
    # States as used for the calculation (results in bypass-factor mode)
    t_in_c: float | None = None
    rh_in_pct: float | None = None
    t_out_c: float | None = None
    rh_out_pct: float | None = None
    # Moisture stream (sensible-capacity mode): room air + delta_x = cooler inlet
    t_room_c: float | None = None        # degC, room air (before moisture is added)
    rh_room_pct: float | None = None
    w_room: float | None = None          # kg/kg
    moisture_kg_h: float | None = None
    delta_x: float | None = None         # kg/kg
    q_moisture: float | None = None      # kW, latent load caused by the moisture


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


def _isenthalp_saturation(h_j: float, p: float):
    """Saturated state on the isenthalp h_j [J/kg dry air]: returns (T [K], w) or (None, None)."""
    try:
        T = HAPropsSI("T", "Hda", h_j, "R", 1.0, "P", p)
        return T, HAPropsSI("W", "T", T, "R", 1.0, "P", p)
    except ValueError:
        return None, None


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
    if abs(m_cond) < 1e-9:  # suppress numerical noise (dry coil: prints -0.00)
        m_cond = 0.0
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

    T_lim, w_lim = _isenthalp_saturation(h_out, p_pa)

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
        t_limit=None if T_lim is None else T_lim - 273.15,
        w_limit=w_lim,
        t_in_c=t_in_c,
        rh_in_pct=rh_in_pct,
        t_out_c=t_out_c,
        rh_out_pct=rh_out_pct,
    )


def _evaporation_enthalpy(t_c: float) -> float:
    """Enthalpy of vaporisation of water [J/kg] at t_c (saturated liquid -> saturated vapour, [3])."""
    T = t_c + 273.15
    return PropsSI("H", "T", T, "Q", 1, "Water") - PropsSI("H", "T", T, "Q", 0, "Water")


def cooling_load_from_capacity(
    q_kw: float,
    t_in_c: float,
    rh_in_pct: float,
    t_out_c: float,
    rh_out_pct: float,
    p_pa: float = P_STD,
    t_condensate_c: float | None = None,
    capacity_kind: str = "total",
    moisture_kg_h: float | None = None,
    moisture_kw: float | None = None,
) -> CoolingResult:
    """Volume flow that a given capacity [kW] can process (outlet T and RH given).

    capacity_kind:
      "total"    - q_kw is the total air-side capacity m * (h_in - h_out).
      "sensible" - q_kw is only the sensible part (cooling at the outlet
                   humidity ratio); the latent part follows from x_in -> x_out.
    moisture_kg_h / moisture_kw (only with "sensible"; give at most one): moisture
    stream added to the room air before the cooler, as mass flow or as
    evaporation power. t_in_c / rh_in_pct are then the ROOM air state; the
    cooler inlet state is returned in the result (t_in_c, rh_in_pct, w_in).
    The volume flow in the result refers to the cooler inlet state.
    """
    if q_kw <= 0.0:
        raise ValueError("Capacity must be positive.")
    if capacity_kind not in ("total", "sensible"):
        raise ValueError(f"Unknown capacity kind: {capacity_kind}")
    if moisture_kg_h is not None and moisture_kw is not None:
        raise ValueError("Give the moisture stream either in kg/h or as evaporation power, not both.")
    if (moisture_kg_h or 0.0) < 0.0 or (moisture_kw or 0.0) < 0.0:
        raise ValueError("The moisture stream must not be negative.")
    has_moisture = bool(moisture_kg_h) or bool(moisture_kw)
    if has_moisture and capacity_kind != "sensible":
        raise ValueError("A moisture stream can only be combined with a purely sensible capacity.")

    T_in, T_out = t_in_c + 273.15, t_out_c + 273.15
    w_room = HAPropsSI("W", "T", T_in, "R", rh_in_pct / 100.0, "P", p_pa)
    w_out = HAPropsSI("W", "T", T_out, "R", rh_out_pct / 100.0, "P", p_pa)

    if capacity_kind == "total":
        h_in = HAPropsSI("Hda", "T", T_in, "W", w_room, "P", p_pa)
        h_out = HAPropsSI("Hda", "T", T_out, "W", w_out, "P", p_pa)
        if h_in <= h_out:
            raise ValueError("Outlet enthalpy must be below the inlet enthalpy (cooling).")
        v_in = HAPropsSI("Vda", "T", T_in, "W", w_room, "P", p_pa)
        m_da = q_kw * 1e3 / (h_in - h_out)
        return cooling_load(m_da * v_in, "m3s", t_in_c, rh_in_pct, t_out_c, rh_out_pct,
                            p_pa=p_pa, flow_ref="in", t_condensate_c=t_condensate_c)

    # ---- sensible capacity: cooling at the outlet humidity ratio -------------------
    dh_sens = (HAPropsSI("Hda", "T", T_in, "W", w_out, "P", p_pa)
               - HAPropsSI("Hda", "T", T_out, "W", w_out, "P", p_pa))
    if dh_sens <= 0.0:
        raise ValueError("For a sensible capacity the outlet temperature must be below the inlet temperature.")
    m_da = q_kw * 1e3 / dh_sens

    # Moisture stream [kg/s]; an evaporation power uses h_fg at the inlet temperature
    if moisture_kw:
        m_w = moisture_kw * 1e3 / _evaporation_enthalpy(t_in_c)
    elif moisture_kg_h:
        m_w = moisture_kg_h / 3600.0
    else:
        m_w = 0.0

    delta_x = m_w / m_da
    w_in = w_room + delta_x
    w_sat_in = HAPropsSI("W", "T", T_in, "R", 1.0, "P", p_pa)
    if w_in > w_sat_in:  # CoolProp cannot evaluate supersaturated states, so check explicitly
        raise ValueError(
            f"The moisture stream would supersaturate the air at {t_in_c:.1f} degC: "
            f"x = {w_in * 1000:.1f} g/kg > saturation {w_sat_in * 1000:.1f} g/kg at the resulting "
            f"air flow of {m_da * 3600:.0f} kg/h dry air. Reduce the moisture stream."
        )
    rh_coil = min(HAPropsSI("R", "T", T_in, "W", w_in, "P", p_pa), 1.0)
    v_in = HAPropsSI("Vda", "T", T_in, "W", w_in, "P", p_pa)

    res = cooling_load(m_da * v_in, "m3s", t_in_c, rh_coil * 100.0, t_out_c, rh_out_pct,
                       p_pa=p_pa, flow_ref="in", t_condensate_c=t_condensate_c)
    if m_w > 0.0:
        res.t_room_c, res.rh_room_pct, res.w_room = t_in_c, rh_in_pct, w_room
        res.moisture_kg_h, res.delta_x = m_w * 3600.0, delta_x
        res.q_moisture = m_da * (HAPropsSI("Hda", "T", T_in, "W", w_in, "P", p_pa)
                                 - HAPropsSI("Hda", "T", T_in, "W", w_room, "P", p_pa)) / 1e3
    return res


def cooling_load_from_bf(
    flow: float | None,
    unit: str,
    t_in_c: float,
    rh_in_pct: float,
    bypass_factor: float,
    p_pa: float = P_STD,
    h_out_kj: float | None = None,
    q_kw: float | None = None,
    t_condensate_c: float | None = None,
) -> CoolingResult:
    """Coil surface temperature and outlet state for a given bypass factor.

    With a volume flow (referring to the inlet state): give exactly one target,
    outlet enthalpy h_out_kj [kJ/kg dry air] or capacity q_kw [kW, air side].
    Without a volume flow (flow=None): give both h_out_kj and q_kw; the volume
    flow at the inlet is then a result.
    Outlet temperature and RH are results (a coil with a given BF has only one
    degree of freedom, the surface temperature).
    """
    n_targets = (h_out_kj is not None) + (q_kw is not None)
    if flow is not None and n_targets != 1:
        raise ValueError("With a given volume flow, give exactly one target: "
                         "outlet enthalpy or capacity.")
    if flow is None and n_targets != 2:
        raise ValueError("Without a volume flow, give both the target outlet enthalpy "
                         "and the capacity.")
    if q_kw is not None and q_kw <= 0.0:
        raise ValueError("Capacity must be positive.")
    if not 0.0 <= bypass_factor < 1.0:
        raise ValueError("Bypass factor must be in the range [0, 1).")

    T_in = t_in_c + 273.15
    w_in = HAPropsSI("W", "T", T_in, "R", rh_in_pct / 100.0, "P", p_pa)
    h_in = HAPropsSI("Hda", "T", T_in, "W", w_in, "P", p_pa)
    v_in = HAPropsSI("Vda", "T", T_in, "W", w_in, "P", p_pa)
    dew_in = HAPropsSI("D", "T", T_in, "W", w_in, "P", p_pa) - 273.15

    if flow is not None:
        m_da = _to_m3s(flow, unit) / v_in
        h_out = h_out_kj * 1e3 if h_out_kj is not None else h_in - q_kw * 1e3 / m_da
        if h_out >= h_in:
            raise ValueError("Target outlet enthalpy must be below the inlet enthalpy (cooling).")
    else:
        h_out = h_out_kj * 1e3
        if h_out >= h_in:
            raise ValueError("Target outlet enthalpy must be below the inlet enthalpy (cooling).")
        m_da = q_kw * 1e3 / (h_in - h_out)  # dry-air mass flow that delivers the capacity

    # Mixing rule: h_out = BF * h_in + (1 - BF) * h_surface
    h_s = (h_out - bypass_factor * h_in) / (1.0 - bypass_factor)
    try:
        T_wet = HAPropsSI("T", "Hda", h_s, "R", 1.0, "P", p_pa)  # saturated state with h_s
        w_wet = HAPropsSI("W", "T", T_wet, "R", 1.0, "P", p_pa)
        wet = w_wet < w_in - 1e-9
        if wet:      # dehumidifying coil: surface state is saturated
            T_surf, w_surf = T_wet, w_wet
            w_out = bypass_factor * w_in + (1.0 - bypass_factor) * w_wet
        else:        # dry coil: constant humidity ratio
            w_out = w_surf = w_in
            T_surf = HAPropsSI("T", "Hda", h_s, "W", w_in, "P", p_pa)
        T_out = HAPropsSI("T", "Hda", h_out, "W", w_out, "P", p_pa)
        rh_out = min(HAPropsSI("R", "Hda", h_out, "W", w_out, "P", p_pa), 1.0)
        if T_surf < 223.15:
            raise ValueError("surface temperature below -50 degC")
    except ValueError as exc:
        raise ValueError(
            "The required surface / outlet state is outside the range CoolProp can evaluate "
            "(target enthalpy too low for this bypass factor)."
        ) from exc

    # All capacities, flows and the ideal-coil limit come from the standard routine.
    res = cooling_load(m_da * v_in, "m3s", t_in_c, rh_in_pct, T_out - 273.15, rh_out * 100.0,
                       p_pa=p_pa, flow_ref="in", t_condensate_c=t_condensate_c)

    res.t_surface = T_surf - 273.15
    res.w_adp = w_surf
    res.bypass_factor = bypass_factor
    res.contact_factor = 1.0 - bypass_factor
    if not wet:
        res.surface_note = (
            f"Dry coil: the surface stays above the inlet dew point ({dew_in:.1f} degC), so no "
            "condensation occurs and x_out = x_in. Surface temperature from the mixing rule "
            "at constant humidity ratio."
        )
    elif res.t_surface < 0.0:
        res.surface_note = "Surface temperature below 0 degC: frost formation on the coil is likely."
    else:
        res.surface_note = ""
    return res


def format_report(r: CoolingResult) -> str:
    """Return a plain-text report (used by the CLI and the GUI)."""
    def state_lines(label, t, rh, w, h, dew):
        head = f"{label}: " + (f"{t:5.2f} degC, {rh:5.1f} % RH" if t is not None else "")
        return [head, f"        x = {w * 1000:5.2f} g/kg, h = {h:6.2f} kJ/kg, dew point = {dew:5.1f} degC"]

    lines = ["--- Air states ---"]
    if r.moisture_kg_h is not None:
        lines.append(f"Room  : {r.t_room_c:5.2f} degC, {r.rh_room_pct:5.1f} % RH, x = {r.w_room * 1000:5.2f} g/kg")
        lines.append(f"        + moisture {r.moisture_kg_h:.2f} kg/h -> dx = {r.delta_x * 1000:.2f} g/kg "
                     f"(latent load {r.q_moisture:.2f} kW)")
    lines += state_lines("Inlet ", r.t_in_c, r.rh_in_pct, r.w_in, r.h_in, r.dew_in)
    lines += state_lines("Outlet", r.t_out_c, r.rh_out_pct, r.w_out, r.h_out, r.dew_out)
    lines += [
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
    if r.t_limit is not None:
        lines.append(f"Ideal-coil limit (BF=0) : {r.t_limit:6.2f} degC  (T_s can not be higher)")
    if r.surface_note:
        lines.append(r.surface_note)

    if r.w_out > r.w_in + 1e-6:
        lines.append("\nWARNING: outlet humidity ratio is higher than inlet -> this needs "
                     "humidification, not just cooling. Check your outlet values.")
    elif r.q_total < 0:
        lines.append("\nWARNING: negative capacity -> this is a heating process.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cooling capacity / volume flow for an air stream (CoolProp).")
    ap.add_argument("--flow", type=float, help="Volume flow (omit it and give --q to compute the flow)")
    ap.add_argument("--unit", default="m3h", choices=["m3h", "ls", "m3s"], help="Flow unit (default: m3h)")
    ap.add_argument("--flow-ref", default="in", choices=["in", "out"],
                    help="State at which the volume flow is specified (default: in; not used with --bf)")
    ap.add_argument("--t-in", type=float, required=True, help="Inlet temperature [degC]")
    ap.add_argument("--rh-in", type=float, required=True, help="Inlet relative humidity [%%]")
    ap.add_argument("--t-out", type=float, help="Outlet temperature [degC]")
    ap.add_argument("--rh-out", type=float, help="Outlet relative humidity [%%]")
    ap.add_argument("--bf", type=float, help="Bypass factor (bypass-factor mode: outlet state is a result; "
                    "needs --h-out and/or --q instead of --t-out/--rh-out)")
    ap.add_argument("--h-out", type=float, help="Target outlet enthalpy [kJ/kg dry air] (bypass mode)")
    ap.add_argument("--q", type=float, help="Cooling capacity [kW, total air side unless --q-kind sensible]")
    ap.add_argument("--q-kind", default="total", choices=["total", "sensible"],
                    help="Meaning of --q: total (sensible + latent) or sensible only (default: total)")
    ap.add_argument("--moisture-kgh", type=float,
                    help="Moisture stream added to the room air [kg/h] (needs --q-kind sensible)")
    ap.add_argument("--moisture-kw", type=float,
                    help="Moisture stream as evaporation power [kW] (needs --q-kind sensible)")
    ap.add_argument("--p", type=float, default=P_STD, help="Pressure [Pa] (default: 101325)")
    ap.add_argument("--t-cond", type=float, default=None, help="Condensate temperature [degC] (default: outlet T)")
    args = ap.parse_args()

    if args.bf is not None and (args.q_kind != "total" or args.moisture_kgh or args.moisture_kw):
        ap.error("--q-kind sensible and the moisture options are not available with --bf")
    if args.bf is not None:
        if args.flow is not None and (args.h_out is None) == (args.q is None):
            ap.error("--bf with --flow needs exactly one of --h-out or --q")
        if args.flow is None and (args.h_out is None or args.q is None):
            ap.error("--bf without --flow needs both --h-out and --q (the flow is then the result)")
        result = cooling_load_from_bf(
            args.flow, args.unit, args.t_in, args.rh_in, args.bf, p_pa=args.p,
            h_out_kj=args.h_out, q_kw=args.q, t_condensate_c=args.t_cond,
        )
    else:
        if args.t_out is None or args.rh_out is None:
            ap.error("--t-out and --rh-out are required (or use --bf with --h-out / --q)")
        if (args.flow is None) == (args.q is None):
            ap.error("give exactly one of --flow or --q (the other one is the result)")
        if args.flow is not None:
            if args.q_kind != "total" or args.moisture_kgh or args.moisture_kw:
                ap.error("--q-kind sensible and the moisture options only work with --q (not --flow)")
            result = cooling_load(
                args.flow, args.unit, args.t_in, args.rh_in, args.t_out, args.rh_out,
                p_pa=args.p, flow_ref=args.flow_ref, t_condensate_c=args.t_cond,
            )
        else:
            result = cooling_load_from_capacity(
                args.q, args.t_in, args.rh_in, args.t_out, args.rh_out,
                p_pa=args.p, t_condensate_c=args.t_cond, capacity_kind=args.q_kind,
                moisture_kg_h=args.moisture_kgh, moisture_kw=args.moisture_kw,
            )
    print(format_report(result))


if __name__ == "__main__":
    main()
