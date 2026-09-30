#!/usr/bin/env python3
"""
Minimal GUI for cooling_load.py: input fields, text results and a Mollier
h-x diagram with the process drawn in.

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

# label -> (key, default)
FIELDS = [
    ("Volume flow", "flow", "2000"),
    ("Inlet temperature [°C]", "t_in", "27"),
    ("Inlet rel. humidity [%]", "rh_in", "50"),
    ("Outlet temperature [°C]", "t_out", "13"),
    ("Outlet rel. humidity [%]", "rh_out", "90"),
    ("Pressure [hPa]", "p", "1013.25"),
]
UNITS = {"m³/h": "m3h", "l/s": "ls"}
FLOW_REFS = {"at inlet state": "in", "at outlet state": "out"}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self._fit_job = None
        self.title("Cooling load calculator")
        self.geometry("1250x720")

        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=4)
        right.pack(side="right", fill="both", expand=True)

        # ---- Inputs ----
        box = ttk.LabelFrame(left, text="Input", padding=8)
        box.pack(fill="x")
        self.vars = {}
        for row, (label, key, default) in enumerate(FIELDS):
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(box, textvariable=var, width=10, justify="right")
            entry.grid(row=row, column=1, pady=2, padx=4)
            entry.bind("<Return>", lambda _e: self.calculate())
            self.vars[key] = var

        self.unit = tk.StringVar(value="m³/h")
        ttk.Combobox(box, textvariable=self.unit, values=list(UNITS), width=6,
                     state="readonly").grid(row=0, column=2, padx=2)
        ttk.Label(box, text="Flow specified").grid(row=len(FIELDS), column=0, sticky="w", pady=2)
        self.flow_ref = tk.StringVar(value="at inlet state")
        ttk.Combobox(box, textvariable=self.flow_ref, values=list(FLOW_REFS), width=14,
                     state="readonly").grid(row=len(FIELDS), column=1, columnspan=2, pady=2, padx=4)

        ttk.Button(left, text="Calculate", command=self.calculate).pack(fill="x", pady=8)

        # ---- Results ----
        res_box = ttk.LabelFrame(left, text="Results", padding=4)
        res_box.pack(fill="both", expand=True)
        self.text = tk.Text(res_box, width=64, height=30, font=("Courier", 9), wrap="word",
                            state="disabled")
        self.text.pack(fill="both", expand=True)

        # ---- Diagram ----
        self.fig = Figure(figsize=(4, 4), dpi=100)  # small request size: the canvas then expands to the free space
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        widget.bind("<Configure>", self._fit_figure, add="+")

        self.calculate()

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

    def _get(self, key: str) -> float:
        return float(self.vars[key].get().replace(",", "."))  # accept decimal comma

    def calculate(self):
        try:
            flow, p_pa = self._get("flow"), self._get("p") * 100.0
            t_in, rh_in = self._get("t_in"), self._get("rh_in")
            t_out, rh_out = self._get("t_out"), self._get("rh_out")
            if not (0 < rh_in <= 100 and 0 < rh_out <= 100):
                raise ValueError("Relative humidity must be in (0, 100] %.")
            if flow <= 0 or p_pa <= 0:
                raise ValueError("Flow and pressure must be positive.")
            res = cl.cooling_load(flow, UNITS[self.unit.get()], t_in, rh_in, t_out, rh_out,
                                  p_pa=p_pa, flow_ref=FLOW_REFS[self.flow_ref.get()])
            adp = (res.t_surface, res.w_adp) if res.t_surface is not None else None
            draw_mollier(self.fig, p_pa, (t_in, res.w_in), (t_out, res.w_out), adp)
            self.canvas.draw()
        except ValueError as err:
            messagebox.showerror("Input error", str(err))
            return

        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", cl.format_report(res))
        self.text.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
