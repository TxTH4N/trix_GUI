"""
Gaussian Peak Fitter (PySide6 + lmfit)

What it does
------------
- Loads a single run using your `import_triX_single` class in `load_triX_murr.py`.
- Lets you choose the **number of Gaussian peaks** to fit plus a **linear background**.
- Uses **lmfit** with **y-error bars as weights** (weights = 1/yerr).
- Shows a **table** of best-fit parameters with **uncertainties** (1σ from the covariance).
- Plots data with error bars, the **total fit**, and each **peak component**.

Setup
-----
pip install pyside6 matplotlib lmfit

Run
---
python fit_gaussian_peaks.py

Notes
-----
- Initial guesses: by default, the GUI auto-guesses peak centers from local maxima
  and amplitudes from data; widths start from 1/20 of x-range. You can switch off
  Auto-guess to enter simple center guesses (comma-separated) if you prefer.
- The lmfit Gaussian uses parameters (amplitude [area], center, sigma). The table
  also reports FWHM = 2*sqrt(2*ln 2)*sigma for convenience.
"""

import sys
import os
from typing import List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox, QGridLayout,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    # Your loader
    from load_triX import import_triX_single  # type: ignore
except Exception:
    class import_triX_single:  # type: ignore
        def __init__(self, instrument: str, exp: int, label_T: str):
            raise RuntimeError("Ensure load_triX_murr.py with class import_triX_single is present.")

from lmfit.models import GaussianModel, LinearModel


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y.copy()
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode='same')


def find_local_maxima(x: np.ndarray, y: np.ndarray) -> List[int]:
    # very simple local max finder
    idxs = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i-1] and y[i] >= y[i+1]:
            idxs.append(i)
    return idxs


def auto_initial_guesses(x: np.ndarray, y: np.ndarray, npeaks: int) -> Tuple[List[float], List[float], List[float]]:
    # Smooth a bit, find local maxima, pick top npeaks
    win = max(3, len(x)//50)
    ys = moving_average(y, win)
    maxima = find_local_maxima(x, ys)
    if not maxima:
        maxima = list(np.argsort(y)[-npeaks:])
    # pick top npeaks by y value
    maxima = sorted(maxima, key=lambda i: y[i], reverse=True)[:npeaks]
    centers = [float(x[i]) for i in maxima]
    # width guess ~ fraction of span
    span = float(np.max(x) - np.min(x)) if len(x) > 1 else 1.0
    sigmas = [max(span/20.0, 1e-6)] * npeaks
    # amplitude (area) guess ~ height * sigma * sqrt(2*pi)
    amps = [float(max(y[i], 0.0)) * sigmas[0] * np.sqrt(2*np.pi) for i in range(len(centers))]
    return amps, centers, sigmas


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), layout='constrained')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, which='both', alpha=0.3, linestyle='--')

    def clear(self):
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, which='both', alpha=0.3, linestyle='--')
        self.fig.canvas.draw_idle()

    def plot_data(self, x, y, yerr):
        self.ax.errorbar(x, y, yerr=yerr, fmt='o', ms=4, lw=1, capsize=2, label='data')

    def plot_fit(self, x, yfit):
        self.ax.plot(x, yfit, lw=2, label='fit')

    def plot_component(self, x, ycomp, label):
        self.ax.plot(x, ycomp, lw=1, alpha=0.8, label=label)

    def refresh(self, title: str = ""):
        if title:
            self.ax.set_title(title)
        self.ax.legend(frameon=False)
        self.fig.canvas.draw_idle()


class FitWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaussian Peak Fitter (lmfit)")
        self.setMinimumSize(1150, 720)

        # --- Controls ---
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose parent folder …")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.on_browse)

        self.instrument_edit = QLineEdit()
        self.instrument_edit.setPlaceholderText("Instrument (e.g., HB3, CTAX, BT7 …)")

        self.exp_spin = QSpinBox(); self.exp_spin.setRange(0, 999999); self.exp_spin.setValue(1); self.exp_spin.setPrefix("exp ")
        self.run_spin = QSpinBox(); self.run_spin.setRange(0, 999999); self.run_spin.setValue(1); self.run_spin.setPrefix("run ")

        self.temp_label_edit = QLineEdit(); self.temp_label_edit.setPlaceholderText("Temperature column header (e.g., T_sample)")
        self.xname_edit = QLineEdit(); self.xname_edit.setPlaceholderText("Override x name (optional)")
        self.norm_check = QCheckBox("Normalize to counts/sec (monitor)"); self.norm_check.setChecked(True)

        self.npeaks_spin = QSpinBox(); self.npeaks_spin.setRange(1, 20); self.npeaks_spin.setValue(1)
        self.auto_guess_check = QCheckBox("Auto-guess initial centers/amplitudes"); self.auto_guess_check.setChecked(True)
        self.centers_edit = QLineEdit(); self.centers_edit.setPlaceholderText("Manual centers (comma-separated), used when Auto-guess is OFF")

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.on_load)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setEnabled(False)
        self.fit_btn.clicked.connect(self.on_fit)

        # --- Plot ---
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Table for parameters ---
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Component", "Amplitude (area)", "Center", "Sigma", "FWHM", "Unc (1σ)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # --- Layout ---
        form = QGridLayout()
        r = 0
        form.addWidget(QLabel("Data folder"), r, 0); form.addWidget(self.path_edit, r, 1); form.addWidget(browse_btn, r, 2); r += 1
        form.addWidget(QLabel("Instrument"), r, 0); form.addWidget(self.instrument_edit, r, 1); r += 1
        form.addWidget(QLabel("Experiment #"), r, 0); form.addWidget(self.exp_spin, r, 1); r += 1
        form.addWidget(QLabel("Run #"), r, 0); form.addWidget(self.run_spin, r, 1); r += 1
        form.addWidget(QLabel("Temp label"), r, 0); form.addWidget(self.temp_label_edit, r, 1); r += 1
        form.addWidget(QLabel("X name (optional)"), r, 0); form.addWidget(self.xname_edit, r, 1); r += 1
        form.addWidget(self.norm_check, r, 1); r += 1
        form.addWidget(QLabel("# Peaks"), r, 0); form.addWidget(self.npeaks_spin, r, 1); r += 1
        form.addWidget(self.auto_guess_check, r, 1); r += 1
        form.addWidget(QLabel("Manual centers"), r, 0); form.addWidget(self.centers_edit, r, 1); r += 1
        form.addWidget(self.load_btn, r, 0); form.addWidget(self.fit_btn, r, 1); r += 1

        top = QWidget(); top_layout = QVBoxLayout(top)
        top_layout.addLayout(form)
        top_layout.addWidget(self.toolbar)
        top_layout.addWidget(self.canvas, 1)
        top_layout.addWidget(QLabel("Fit parameters"))
        top_layout.addWidget(self.table)

        self.setCentralWidget(top)
        self.setStatusBar(QStatusBar())

        # State
        self._data = None  # tuple (x, y, yerr)
        self._label = None

    # --- Actions ---
    def on_browse(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose data root folder")
        if directory:
            self.path_edit.setText(directory)

    def on_load(self):
        try:
            folder = self.path_edit.text().strip()
            instrument = self.instrument_edit.text().strip()
            exp = int(self.exp_spin.value())
            run = int(self.run_spin.value())
            temp_label = self.temp_label_edit.text().strip()
            x_override = self.xname_edit.text().strip() or None
            norm = self.norm_check.isChecked()

            if not folder or not os.path.isdir(folder):
                raise ValueError("Please choose a valid folder.")
            if not instrument:
                raise ValueError("Instrument cannot be empty.")
            if not temp_label:
                raise ValueError("Temperature label cannot be empty.")

            loader = import_triX_single(instrument=instrument, exp=exp, label_T=temp_label)
            label, x, y, yerr = loader.load_data(folder, run, nor_to_cps=norm, name_x=x_override)
            self._data = (x, y, yerr)
            self._label = label

            self.canvas.clear()
            self.canvas.plot_data(x, y, yerr)
            title = f"{label.get('samplename','')}  T={label.get('temperature','?')}±{label.get('tem_error','?')} K"
            self.canvas.refresh(title=title)
            self.statusBar().showMessage(f"Loaded {len(x)} points", 4000)
            self.fit_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def _build_model(self, x: np.ndarray, npeaks: int, auto_guess: bool, manual_centers: Optional[List[float]]):
        # Linear background + sum of Gaussians
        model = LinearModel(prefix='bg_')
        params = model.make_params(intercept=float(np.median(x*0)), slope=0.0)

        for i in range(npeaks):
            g = GaussianModel(prefix=f"g{i}_")
            model = model + g

        # initial guesses
        if auto_guess:
            amps, centers, sigmas = auto_initial_guesses(x, self._data[1], npeaks)
        else:
            if not manual_centers or len(manual_centers) < npeaks:
                raise ValueError("Provide at least as many manual centers as #peaks.")
            centers = manual_centers[:npeaks]
            span = float(np.max(x) - np.min(x)) if len(x) > 1 else 1.0
            sigmas = [max(span/20.0, 1e-6)] * npeaks
            # amplitude guess from local y near center
            amps = []
            for c in centers:
                idx = int(np.clip(np.searchsorted(x, c), 0, len(x)-1))
                amps.append(float(max(self._data[1][idx], 0.0) * sigmas[0] * np.sqrt(2*np.pi)))

        # set parameters with reasonable bounds
        for i in range(npeaks):
            g = GaussianModel(prefix=f"g{i}_")
            p = g.make_params(amplitude=amps[i], center=centers[i], sigma=sigmas[i])
            # bounds
            p[f"g{i}_sigma"].min = 1e-6
            p[f"g{i}_sigma"].max = max(np.max(x) - np.min(x), 1e6)
            p[f"g{i}_center"].min = float(np.min(x))
            p[f"g{i}_center"].max = float(np.max(x))
            params.update(p)

        return model, params

    def on_fit(self):
        try:
            if self._data is None:
                raise ValueError("Load data first.")
            x, y, yerr = self._data
            npeaks = int(self.npeaks_spin.value())
            auto_guess = self.auto_guess_check.isChecked()
            manual_centers = None
            if not auto_guess:
                txt = self.centers_edit.text().strip()
                manual_centers = [float(s) for s in txt.split(',') if s.strip()]

            model, params = self._build_model(x, npeaks, auto_guess, manual_centers)

            # weights = 1/yerr (avoid div-by-zero)
            yerr_safe = np.where((yerr is not None) & (yerr > 0), yerr, np.nan)
            if np.any(~np.isfinite(yerr_safe)):
                # replace non-finite with median positive
                pos = yerr[np.where(yerr > 0)]
                med = float(np.median(pos)) if len(pos) else 1.0
                yerr_safe = np.where(np.isfinite(yerr) & (yerr > 0), yerr, med)
            weights = 1.0 / yerr_safe

            result = model.fit(y, params, x=x, weights=weights)

            # Plot
            self.canvas.clear()
            self.canvas.plot_data(x, y, yerr)
            self.canvas.plot_fit(x, result.best_fit)
            # Components
            comps = result.eval_components(x=x)
            for name, ycomp in comps.items():
                # name like 'g0_' or 'bg_'
                if name.startswith('g'):
                    label = f"{name[:-1]} component"
                else:
                    label = name[:-1]
                self.canvas.plot_component(x, ycomp, label)
            self.canvas.refresh(title="Gaussian fit with linear background")

            # Table
            self._fill_table_from_result(result, npeaks)

            self.statusBar().showMessage(f"Fit complete. Redchi={result.redchi:.3g}", 6000)
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))

    def _fill_table_from_result(self, result, npeaks: int):
        # Columns: Component, Amplitude, Center, Sigma, FWHM, Unc
        # We will show 1 row per Gaussian and 1 row for background
        rows = npeaks + 1
        self.table.setRowCount(rows)

        # Background row
        try:
            b_slope = result.params['bg_slope']
            b_inter = result.params['bg_intercept']
            self._set_row(0, "bg", "—", "—", "—", f"slope={b_slope.value:.3g}±{b_slope.stderr or 0:.2g}, intercept={b_inter.value:.3g}±{b_inter.stderr or 0:.2g}")
        except Exception:
            self._set_row(0, "bg", "—", "—", "—", "(not available)")

        # Peak rows
        for i in range(npeaks):
            try:
                amp = result.params[f'g{i}_amplitude']
                cen = result.params[f'g{i}_center']
                sig = result.params[f'g{i}_sigma']
                fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig.value

                amp_s = amp.stderr if amp.stderr is not None else 0.0
                cen_s = cen.stderr if cen.stderr is not None else 0.0
                sig_s = sig.stderr if sig.stderr is not None else 0.0
                fwhm_s = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig_s

                self._set_row(
                    i+1,
                    f"g{i}",
                    f"{amp.value:.6g} ± {amp_s:.2g}",
                    f"{cen.value:.6g} ± {cen_s:.2g}",
                    f"{sig.value:.6g} (σ) / {fwhm:.6g} (FWHM)\n± {sig_s:.2g} / {fwhm_s:.2g}",
                    "—",
                )
            except Exception as e:
                self._set_row(i+1, f"g{i}", "—", "—", "—", f"(error: {e})")

    def _set_row(self, row: int, comp: str, amp: str, cen: str, sig_fwhm: str, extra: str):
        items = [comp, amp, cen, sig_fwhm, "", extra]
        for col, text in enumerate(items):
            it = QTableWidgetItem(text)
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, it)


def main():
    app = QApplication(sys.argv)
    w = FitWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
