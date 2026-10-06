#!/usr/bin/env python3
"""
Minimal GUI for cooling_load.py: input fields, text results and a Mollier
h-x diagram with the process drawn in.

Known quantity : volume flow (-> capacity is the result) or capacity
                 (-> volume flow is the result).
State mode     : outlet T / RH given (-> coil surface temperature and bypass
                 factor are results), or bypass factor given with a target
                 (-> outlet T / RH and surface temperature are results).

Requires: CoolProp, matplotlib (tkinter ships with most Python installers;
on Debian/Ubuntu: sudo apt install python3-tk).

Run: python cooling_gui.py
"""

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import cooling_load as cl
from mollier import draw_mollier

KNOWN = {"Volume flow": "flow", "Cooling capacity": "q"}
UNITS = {"flow": {"m³/h": "m3h", "l/s": "ls"}, "q": {"kW": "kw"}}
VALUE_DEFAULTS = {"flow": "2000", "q": "15"}
FLOW_REFS = {"at inlet state": "in", "at outlet state": "out"}
TARGETS = {"Outlet enthalpy [kJ/kg]": "h", "Capacity [kW]": "q"}
CAP_KINDS = {"Total (sens. + latent)": "total", "Sensible only": "sensible"}
MOISTURE_UNITS = {"kg/h": "kgh", "kW (evaporation)": "kw"}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self._fit_job = None
        self.title("Cooling load calculator")
        self.geometry("1250x860")

        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=4)
        right.pack(side="right", fill="both", expand=True)

        box = ttk.LabelFrame(left, text="Input", padding=8)
        box.pack(fill="x")
        self.vars, self.entries = {}, {}
        self.mode = tk.StringVar(value="outlet")
        self.known = tk.StringVar(value="Volume flow")
        self.unit = tk.StringVar(value="m³/h")
        self.flow_ref = tk.StringVar(value="at inlet state")
        self.target_kind = tk.StringVar(value="Outlet enthalpy [kJ/kg]")
        self.cap_kind = tk.StringVar(value="Total (sens. + latent)")
        self.moisture_unit = tk.StringVar(value="kg/h")
        self._known_kind = "flow"
        self._value_store = dict(VALUE_DEFAULTS)

        def label(row, text):
            ttk.Label(box, text=text).grid(row=row, column=0, sticky="w", pady=2)

        def entry(row, key, default):
            var = tk.StringVar(value=default)
            ent = ttk.Entry(box, textvariable=var, width=10, justify="right")
            ent.grid(row=row, column=1, pady=2, padx=4)
            ent.bind("<Return>", lambda _e: self.calculate())
            self.vars[key], self.entries[key] = var, ent

        def combo(row, var, values, width, column=1, span=1, command=None):
            cb = ttk.Combobox(box, textvariable=var, values=list(values), width=width, state="readonly")
            cb.grid(row=row, column=column, columnspan=span, pady=2, padx=4)
            if command:
                cb.bind("<<ComboboxSelected>>", lambda _e: command())
            return cb

        # Known quantity
        label(0, "Known quantity")
        combo(0, self.known, KNOWN, 16, span=2, command=self._update_known)
        label(1, "Value")
        entry(1, "value", VALUE_DEFAULTS["flow"])
        self.unit_cb = combo(1, self.unit, UNITS["flow"], 6, column=2)
        # Inlet state / pressure
        label(2, "Inlet temperature [°C]")
        entry(2, "t_in", "27")
        label(3, "Inlet rel. humidity [%]")
        entry(3, "rh_in", "50")
        label(4, "Pressure [hPa]")
        entry(4, "p", "1013.25")
        label(5, "Flow specified")
        self.flow_ref_cb = combo(5, self.flow_ref, FLOW_REFS, 14, span=2)

        label(6, "Capacity type")
        self.cap_cb = combo(6, self.cap_kind, CAP_KINDS, 20, span=2, command=self._update_mode)
        label(7, "Moisture stream")
        entry(7, "moisture", "0")
        self.moisture_cb = combo(7, self.moisture_unit, MOISTURE_UNITS, 14, column=2)

        ttk.Separator(box).grid(row=8, column=0, columnspan=3, sticky="ew", pady=6)

        # Mode A: outlet state given
        ttk.Radiobutton(box, text="Outlet state given", variable=self.mode, value="outlet",
                        command=self._update_mode).grid(row=9, column=0, columnspan=3, sticky="w")
        label(10, "Outlet temperature [°C]")
        entry(10, "t_out", "13")
        label(11, "Outlet rel. humidity [%]")
        entry(11, "rh_out", "90")

        # Mode B: bypass factor given
        ttk.Radiobutton(box, text="Bypass factor given", variable=self.mode, value="bf",
                        command=self._update_mode).grid(row=12, column=0, columnspan=3, sticky="w", pady=(6, 0))
        label(13, "Bypass factor BF [-]")
        entry(13, "bf", "0.15")
        label(14, "Target")
        self.target_cb = combo(14, self.target_kind, TARGETS, 22, span=2, command=self._update_mode)
        label(15, "Target value")
        entry(15, "target", "36")

        ttk.Button(left, text="Calculate", command=self.calculate).pack(fill="x", pady=8)

        res_box = ttk.LabelFrame(left, text="Results", padding=4)
        res_box.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(res_box, orient="vertical")
        self.text = tk.Text(res_box, width=64, height=26, font=("Courier", 9), wrap="word",
                            state="disabled", yscrollcommand=scroll.set)
        scroll.configure(command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        # Figure: the canvas expands into the free window space
        self.fig = Figure(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        widget.bind("<Configure>", self._fit_figure, add="+")

        self._update_mode()
        self.calculate()

    # ---- widget state --------------------------------------------------------------
    def _set_enabled(self, widget, enabled: bool):
        widget.configure(state="normal" if enabled else "disabled")

    def _update_known(self):
        """Switch between 'volume flow known' and 'capacity known'."""
        new_kind = KNOWN[self.known.get()]
        if new_kind != self._known_kind:
            self._value_store[self._known_kind] = self.vars["value"].get()
            self.vars["value"].set(self._value_store[new_kind])
            self._known_kind = new_kind
            self.unit_cb.configure(values=list(UNITS[new_kind]))
            self.unit.set(next(iter(UNITS[new_kind])))
        self._update_mode()

    def _update_mode(self):
        mode, known = self.mode.get(), self._known_kind
        for key in ("t_out", "rh_out"):
            self._set_enabled(self.entries[key], mode == "outlet")
        for key in ("bf", "target"):
            self._set_enabled(self.entries[key], mode == "bf")
        if mode == "bf" and known == "q":
            # capacity is already given -> the target must be the outlet enthalpy
            self.target_kind.set("Outlet enthalpy [kJ/kg]")
        self.target_cb.configure(state="readonly" if (mode == "bf" and known == "flow") else "disabled")
        # Capacity type / moisture stream: only with a given capacity and outlet T/RH
        cap_free = known == "q" and mode == "outlet"
        if not cap_free:
            self.cap_kind.set("Total (sens. + latent)")
        self.cap_cb.configure(state="readonly" if cap_free else "disabled")
        moisture_ok = cap_free and CAP_KINDS[self.cap_kind.get()] == "sensible"
        self._set_enabled(self.entries["moisture"], moisture_ok)
        self.moisture_cb.configure(state="readonly" if moisture_ok else "disabled")
        self.flow_ref_cb.configure(state="readonly" if (mode == "outlet" and known == "flow") else "disabled")
        self.unit_cb.configure(state="readonly" if known == "flow" else "disabled")

    def _fit_figure(self, _event=None):
        """Debounced: fit the figure to the widget once the backend has settled its DPI."""
        if self._fit_job is not None:
            self.after_cancel(self._fit_job)
        self._fit_job = self.after(50, self._refit)

    def _refit(self):
        """Keep the figure's pixel size equal to the widget size.

        With display scaling (DPI != 100) the Tk backend computes the new size
        with a stale DPI, so the figure sticks out and clips the labels.
        """
        self._fit_job = None
        widget = self.canvas.get_tk_widget()
        dpi = self.fig.dpi
        self.fig.set_size_inches(widget.winfo_width() / dpi, widget.winfo_height() / dpi,
                                 forward=False)
        self.canvas.draw()

    # ---- calculation -------------------------------------------------------------------
    def _get(self, key: str) -> float:
        return float(self.vars[key].get().replace(",", "."))  # accept decimal comma

    def _compute(self):
        known, mode = self._known_kind, self.mode.get()
        value, p_pa = self._get("value"), self._get("p") * 100.0
        t_in, rh_in = self._get("t_in"), self._get("rh_in")
        if value <= 0 or p_pa <= 0:
            raise ValueError("Flow / capacity and pressure must be positive.")
        if not 0 < rh_in <= 100:
            raise ValueError("Relative humidity must be in (0, 100] %.")

        if mode == "outlet":
            t_out, rh_out = self._get("t_out"), self._get("rh_out")
            if not 0 < rh_out <= 100:
                raise ValueError("Relative humidity must be in (0, 100] %.")
            if known == "flow":
                res = cl.cooling_load(value, UNITS["flow"][self.unit.get()], t_in, rh_in, t_out, rh_out,
                                      p_pa=p_pa, flow_ref=FLOW_REFS[self.flow_ref.get()])
            else:
                kind = CAP_KINDS[self.cap_kind.get()]
                moisture = self._get("moisture") if kind == "sensible" else 0.0
                moist_kgh = moisture if (moisture > 0 and MOISTURE_UNITS[self.moisture_unit.get()] == "kgh") else None
                moist_kw = moisture if (moisture > 0 and MOISTURE_UNITS[self.moisture_unit.get()] == "kw") else None
                res = cl.cooling_load_from_capacity(
                    value, t_in, rh_in, t_out, rh_out, p_pa=p_pa, capacity_kind=kind,
                    moisture_kg_h=moist_kgh, moisture_kw=moist_kw)
        else:
            bf, target = self._get("bf"), self._get("target")
            if known == "flow":
                kind = TARGETS[self.target_kind.get()]
                res = cl.cooling_load_from_bf(
                    value, UNITS["flow"][self.unit.get()], t_in, rh_in, bf, p_pa=p_pa,
                    h_out_kj=target if kind == "h" else None,
                    q_kw=target if kind == "q" else None)
            else:
                res = cl.cooling_load_from_bf(None, "m3h", t_in, rh_in, bf, p_pa=p_pa,
                                              h_out_kj=target, q_kw=value)
        return res, p_pa

    def calculate(self):
        try:
            res, p_pa = self._compute()
            adp = (res.t_surface, res.w_adp) if res.t_surface is not None else None
            limit = (res.t_limit, res.w_limit) if res.t_limit is not None else None
            dry = res.w_adp is not None and abs(res.w_adp - res.w_in) < 1e-9
            room = (res.t_room_c, res.w_room) if res.w_room is not None else None
            draw_mollier(self.fig, p_pa, (res.t_in_c, res.w_in), (res.t_out_c, res.w_out),
                         adp, limit, adp_label="Surface" if dry else "ADP", state_room=room)
            self.canvas.draw()
        except ValueError as err:
            messagebox.showerror("Input error", str(err))
            return

        summary = []
        if self._known_kind == "q":
            summary.append(f"=> Volume flow: {res.V_in * 3600:.0f} m3/h = {res.V_in * 1000:.0f} l/s (cooler inlet)")
        else:
            summary.append(f"=> Cooling capacity: {res.q_total:.2f} kW (total, air side)")
        if self._known_kind == "q" and self.cap_kind.get() != "Total (sens. + latent)":
            summary.append(f"=> Total coil capacity: {res.q_total:.2f} kW "
                           f"(sensible {res.q_sensible:.2f} + latent {res.q_latent:.2f})")
        if res.t_surface is not None:
            summary.append(f"=> Coil surface temperature: {res.t_surface:.2f} degC")
        if self.mode.get() == "bf":
            summary.append(f"=> Outlet: {res.t_out_c:.2f} degC, {res.rh_out_pct:.1f} % RH")

        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(summary) + "\n\n" + cl.format_report(res))
        self.text.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
