"""
utils/ui_components.py
========================
Reusable Tkinter widget components.
(Stubs used by main_app.py — expand as needed.)
"""

import tkinter as tk


class StatusBadge(tk.Label):
    """A coloured pill badge showing a status string."""
    COLORS = {
        "NORMAL":  ("#E8F5E9", "#2DC653"),
        "DROWSY":  ("#FFEBEE", "#E63946"),
        "YAWNING": ("#FFF3E0", "#FD7E14"),
        "IDLE":    ("#F5F5F5", "#9E9E9E"),
    }

    def __init__(self, parent, status="IDLE", **kwargs):
        bg, fg = self.COLORS.get(status, ("#F5F5F5", "#9E9E9E"))
        super().__init__(parent, text=status, bg=bg, fg=fg,
                         font=("Segoe UI", 9, "bold"),
                         padx=8, pady=3, **kwargs)

    def update_status(self, status: str):
        bg, fg = self.COLORS.get(status, ("#F5F5F5", "#9E9E9E"))
        self.configure(text=status, bg=bg, fg=fg)


class AlertBanner(tk.Frame):
    """Full-width dismissible alert banner."""

    def __init__(self, parent, message="", level="danger", **kwargs):
        colors = {
            "danger":  "#E63946",
            "warning": "#FD7E14",
            "info":    "#0D6EFD",
        }
        bg = colors.get(level, "#E63946")
        super().__init__(parent, bg=bg, **kwargs)
        self._label = tk.Label(self, text=message,
                               font=("Segoe UI", 12, "bold"),
                               bg=bg, fg="white")
        self._label.pack(side="left", padx=16, pady=10)

    def show(self, message: str):
        self._label.configure(text=message)
        self.pack(fill="x")

    def hide(self):
        self.pack_forget()


class MetricCard(tk.Frame):
    """Compact metric display card."""

    def __init__(self, parent, title: str, value: str = "—",
                 subtitle: str = "", **kwargs):
        super().__init__(parent, bg="#FFFFFF", padx=16, pady=12, **kwargs)
        tk.Label(self, text=title,
                 font=("Segoe UI", 9, "bold"),
                 bg="#FFFFFF", fg="#ADB5BD").pack(anchor="w")
        self._value_lbl = tk.Label(self, text=value,
                                   font=("Segoe UI", 26, "bold"),
                                   bg="#FFFFFF", fg="#1A1A2E")
        self._value_lbl.pack(anchor="w", pady=(4, 2))
        tk.Label(self, text=subtitle,
                 font=("Segoe UI", 9),
                 bg="#FFFFFF", fg="#6C757D").pack(anchor="w")

    def set_value(self, value: str, color: str = "#1A1A2E"):
        self._value_lbl.configure(text=value, fg=color)
