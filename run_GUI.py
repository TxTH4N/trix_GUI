"""
Written by Tianxiong Han
2025.08.24
A GUI for neutron triple axix spectroscopy to view and fit the lines.
"""
import sys
import os
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox, QGridLayout,
    QHBoxLayout, QVBoxLayout, QStatusBar, QColorDialog,QDialog, QDialogButtonBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView,QSplitter
)

# Matplotlib Qt canvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np
from lmfit.models import GaussianModel, LinearModel

# === Import the user's loading class ===
try:
    from load_triX import import_triX_single
except Exception as e:  # pragma: no cover
    import traceback
    print("WARNING: Could not import import_triX_single from load_triX_murr.py.,"
          "Ensure the file exists next to this GUI and defines that class.")
    traceback.print_exc()
    class import_triX_single:
        def __init__(self, instrument: str, exp: int, label_T: str):
            raise RuntimeError(
                "Could not import your loader. Ensure load_triX_murr.py with class import_triX_single exists.")

def parse_runs(spec: str) -> List[int]:
    """Parse comma/range run spec like '1,3,5-8' -> [1,3,5,6,7,8]."""
    out: List[int] = []
    spec = spec.strip()
    if not spec:
        return out
    if '(' not in spec:
        for chunk in spec.split(','):
            c = chunk.strip()
            if not c:
                continue
            if '-' in c:
                a, b = c.split('-', 1)
                a = a.strip(); b = b.strip()
                if not (a.isdigit() and b.isdigit()):
                    raise ValueError(f"Invalid run range: '{c}'")
                start, end = int(a), int(b)
                if end < start:
                    start, end = end, start
                out.extend(range(start, end + 1))
            else:
                if not c.isdigit():
                    raise ValueError(f"Invalid run number: '{c}'")
                out.append(int(c))
    else:
        try:
            a,b,s = [c.strip("()") for c in spec.split(',')]
        except:
            raise ValueError(f"Invalid run number: '{spec}'")
        start, end, step = int(a), int(b), int(s)
        if end > start:
            out.extend(range(start, end + 1, step))
        else:
            raise ValueError(f"Start {start} must be less than end {end}")

    # de-dup preserve order
    seen = set(); uniq = []
    for r in out:
        if r not in seen:
            uniq.append(r); seen.add(r)
    return uniq


def parse_colors(spec: str, n: int) -> List[Optional[str]]:
    """Parse comma-separated colors to length n. Accepts empty for defaults."""
    out: List[Optional[str]] = [None] * n
    if not spec.strip():
        return out

    parts = [p.strip() for p in spec.split(',')]
    idx=0
    for part in parts:
        if idx >= n:
            break
        elif '*' in part:
            nums, color = part.split('*')
            try:
                nums = int(nums)
            except:
                raise ValueError(f"Invalid color format: '{spec}', use format of number*color")
            out[idx:idx+nums] = [color]*nums
            idx += nums
        else:
            out[idx] = part
            idx += 1

    return out


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), layout='constrained')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, which='both', alpha=0.3, linestyle='--')

    def plot_xy(self, x, y, yerr=None, label: Optional[str] = None, color: Optional[str] = None):
        kws = dict(label=label)
        if color is not None:
            kws['color'] = color
        if yerr is not None:
            self.ax.errorbar(x, y, yerr=yerr, fmt='o', ms=4, lw=1, capsize=2, **kws)
        else:
            self.ax.plot(x, y, 'o-', ms=4, lw=1, **kws)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, which='both', alpha=0.3, linestyle='--')

    def refresh(self, title: str = "",xlabel:str="X"):
        if title:
            self.ax.set_title(title)
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(frameon=False)
        self.fig.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriX Data Loader")
        # self.setMinimumSize(1100, 680)
        self.resize(800, 1000)

        # Widgets
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose parent folder that contains /expXXXX/Datafiles …")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.on_browse)

        self.instrument_edit = QLineEdit()
        self.instrument_edit.setPlaceholderText("e.g., HB1A, CG4C, TRIAX...")

        self.exp_spin = QSpinBox()
        self.exp_spin.setRange(0, 999999)
        self.exp_spin.setValue(1)
        self.exp_spin.setPrefix("exp ")

        # Multi-run entry
        self.run_edit = QLineEdit()
        self.run_edit.setPlaceholderText("Run(s): e.g. 12, 14, 18-21, (7, 15, 2)")

        self.temp_label_edit = QLineEdit()
        self.temp_label_edit.setPlaceholderText("Temperature column header in file (e.g., T_sample)")

        self.colors_edit = QLineEdit()
        self.colors_edit.setPlaceholderText("Colors (optional): e.g. C0, 3*C1, #1f77b4, red, #00aa55, black")
        pick_colors_btn = QPushButton("Pick Colors…")
        pick_colors_btn.clicked.connect(self.on_pick_colors)

        self.xname_edit = QLineEdit()
        self.xname_edit.setPlaceholderText("Override x name (optional)")

        self.norm_check = QCheckBox("Normalize to counts/sec (monitor)")
        self.norm_check.setChecked(True)

        self.clear_check = QCheckBox("Clear plot before loading")
        self.clear_check.setChecked(True)

        self.load_btn = QPushButton("Load & Overplot")
        self.load_btn.clicked.connect(self.on_load)

        self.save_fig_btn = QPushButton("Save Figure…")
        self.save_fig_btn.clicked.connect(self.on_save_fig)
        self.save_fig_btn.setEnabled(False)

        self.fit_btn = QPushButton("Fit Peaks…")
        self.fit_btn.clicked.connect(self.on_fit_button)
        self.fit_btn.setEnabled(False)

        # Canvas + toolbar
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout
        form = QGridLayout()
        r = 0
        form.addWidget(QLabel("Data folder"), r, 0)
        form.addWidget(self.path_edit, r, 1)
        form.addWidget(browse_btn, r, 2)
        r += 1
        form.addWidget(QLabel("Instrument"), r, 0)
        form.addWidget(self.instrument_edit, r, 1)
        r += 1
        form.addWidget(QLabel("Experiment #"), r, 0)
        form.addWidget(self.exp_spin, r, 1)
        r += 1
        form.addWidget(QLabel("Temp label"), r, 0)
        form.addWidget(self.temp_label_edit, r, 1)
        r += 1
        form.addWidget(QLabel("Run numbers"), r, 0)
        form.addWidget(self.run_edit, r, 1)
        r += 1
        form.addWidget(QLabel("Colors (optional)"), r, 0)
        form.addWidget(self.colors_edit, r, 1)
        form.addWidget(pick_colors_btn, r, 2)
        r += 1
        form.addWidget(QLabel("X name (optional)"), r, 0)
        form.addWidget(self.xname_edit, r, 1)
        r += 1
        form.addWidget(self.norm_check, r, 1)
        form.addWidget(self.clear_check, r, 2)
        r += 1
        form.addWidget(self.load_btn, r, 2)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        # top_layout.addLayout(form)
        # top_layout.addWidget(self.toolbar)
        # top_layout.addWidget(self.canvas, 1)
        #
        # self.fit_table = QTableWidget(0, 5, self)
        # self.fit_table.setHorizontalHeaderLabels(["Component", "Amplitude (area)", "Center", "Sigma", "FWHM"])
        # self.fit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # top_layout.addWidget(self.fit_table)
        # ---- Controls panel----
        controls = QWidget(self)
        controls.setLayout(form)
        # ---- Plot panel (toolbar + canvas) ----
        plot_panel = QWidget(self)
        plot_v = QVBoxLayout(plot_panel)
        plot_v.setContentsMargins(0, 0, 0, 0)
        plot_v.addWidget(self.toolbar)
        plot_v.addWidget(self.canvas, 1)
        # ---- Fit results table ----
        self.fit_table = QTableWidget(0, 5, self)
        self.fit_table.setHorizontalHeaderLabels(["Component", "Amplitude (area)", "Center", "Sigma", "FWHM"])
        self.fit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.fit_table.setMaximumHeight(100)
        # ---- Vertical splitter with starting ratio (controls : plot : table) ----
        splitter = QSplitter(Qt.Vertical, self)
        splitter.addWidget(controls)
        splitter.addWidget(plot_panel)
        splitter.addWidget(self.fit_table)
        splitter.setSizes([200, 520, 120])
        # Make the plot prefer to grow/shrink
        splitter.setStretchFactor(0, 0)  # controls
        splitter.setStretchFactor(1, 1)  # plot gets the stretch
        splitter.setStretchFactor(2, 0)  # table
        top_layout.addWidget(splitter)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.save_fig_btn)
        button_row.addWidget(self.fit_btn)
        top_layout.addLayout(button_row)

        self.setCentralWidget(top)
        self.setStatusBar(QStatusBar())

        # State
        self._last_loaded = None  # type: Optional[dict]
        self._data_by_run: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}
        self._xlabel_current = "X"

    # --- UI Actions ---
    def on_browse(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose data root folder")
        if directory:
            self.path_edit.setText(directory)

    def on_pick_colors(self):
        try:
            runs = parse_runs(self.run_edit.text().strip())
            if not runs:
                QMessageBox.information(self, "Pick Colors", "Enter run numbers first.")
                return
            chosen = []
            for r in runs:
                c = QColorDialog.getColor(parent=self, title=f"Color for run {r:04d}")
                if c.isValid():
                    chosen.append(c.name())  # hex string like #RRGGBB
                else:
                    chosen.append("")
            self.colors_edit.setText(
                ", ".join(chosen)
            )
        except Exception as e:
            QMessageBox.critical(self, "Color picker error", str(e))

    def on_load(self):
        try:
            folder = self.path_edit.text().strip()
            instrument = self.instrument_edit.text().strip()
            exp = int(self.exp_spin.value())
            run_spec = self.run_edit.text().strip()
            color_spec = self.colors_edit.text().strip()
            temp_label = self.temp_label_edit.text().strip()
            x_override = self.xname_edit.text().strip() or None
            norm = self.norm_check.isChecked()

            if not folder or not os.path.isdir(folder):
                raise ValueError("Please choose a valid folder.")
            if not instrument:
                raise ValueError("Instrument cannot be empty.")
            if not temp_label:
                raise ValueError("Temperature label cannot be empty (must match the file header).")
            runs = parse_runs(run_spec)
            if not runs:
                raise ValueError("Please enter at least one run number (e.g. '12, 14, 18-21').")

            colors = parse_colors(color_spec, len(runs))
            if self.clear_check.isChecked():
                self.canvas.ax.clear()
                self._data_by_run.clear()

            self.statusBar().showMessage("Loading…", 3000)
            loader = import_triX_single(instrument=instrument, exp=exp, label_T=temp_label)
            last_label_for_title = None

            for i, r in enumerate(runs):
                label, x, y, yerr = loader.load_data(folder, r, nor_to_cps=norm, name_x=x_override)
                run_lbl = f"run {r:04d}"
                self.canvas.plot_xy(x, y, yerr=yerr, label='{} - {} K'.format(run_lbl,label['temperature']), color=colors[i])
                last_label_for_title = label
                # cache data for fit
                self._data_by_run[r] = (np.asarray(x), np.asarray(y), np.asarray(yerr), label)
                self._xlabel_current = label.get('x', 'X')

            title = ""
            if last_label_for_title:
                # title = f"{last_label_for_title.get('samplename','')}  T={last_label_for_title.get('temperature','?')}±{last_label_for_title.get('tem_error','?')} K"
                title = f"{last_label_for_title.get('samplename', '')} Run(s) {run_spec}"
            # self.canvas.refresh(title=title,xlabel=label['x'])
            self.canvas.refresh(title=title, xlabel=self._xlabel_current)

            self._last_loaded = {
                'instrument': instrument, 'exp': exp, 'runs': runs,
                'norm': norm, 'folder': folder, 'label': last_label_for_title,
                'colors': colors
            }
            self.statusBar().showMessage(
                f"Loaded runs: {', '.join(f'{r:04d}' for r in runs)} from exp{exp:04d}", 5000)
            self.save_fig_btn.setEnabled(True)
            self.fit_btn.setEnabled(True if self._data_by_run else False)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            self.statusBar().showMessage("Load failed", 5000)

    def on_save_fig(self):
        if not self._last_loaded:
            return
        runs = self._last_loaded['runs']
        suggested = f"triX_exp{self._last_loaded['exp']:04d}_runs{'-'.join(str(r) for r in runs)}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", suggested, "PNG (*.png);;PDF (*.pdf)")
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".pdf":
                self.canvas.fig.savefig(path, bbox_inches='tight')
            else:
                self.canvas.fig.savefig(path, dpi=200, bbox_inches='tight')
            self.statusBar().showMessage(f"Saved figure to {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def on_fit_button(self):
        if not self._data_by_run:
            QMessageBox.information(self, "Fit Peaks", "Load data first.")
            return
        run_ids = sorted(self._data_by_run.keys())
        dlg = FitDialog(self, run_ids)
        if dlg.exec():
            run_id, npeaks, auto, show_components, centers = dlg.values()
            self._fit_run(run_id, npeaks, auto, show_components, centers)

    def _fit_run(self, run_id: int, npeaks: int, auto_guess: bool, show_components: bool,
                 centers: Optional[List[float]] = None):
        try:
            if run_id not in self._data_by_run:
                raise ValueError(f"Run {run_id:04d} not loaded.")
            x, y, yerr, label = self._data_by_run[run_id]

            # Linear background + sum of Gaussians
            model = LinearModel(prefix='bg_')
            params = model.make_params(intercept=float(np.median(y)), slope=0.0)

            span = float(np.max(x) - np.min(x)) if len(x) > 1 else 1.0
            sig0 = max(span / 20.0, 1e-6)

            if auto_guess and not (centers and len(centers) >= 1):
                idxs = np.argsort(y)[-npeaks:]
                centers = list(np.sort(x[idxs]))
                amps = [max(y[i], 0.0) * sig0 * np.sqrt(2 * np.pi) for i in idxs]
            else:
                if centers is None or len(centers) < npeaks:
                    raise ValueError("Please provide at least as many centers as # of Gaussians.")
                centers = centers[:npeaks]
                amps = []
                for c in centers:
                    idx = int(np.clip(np.searchsorted(x, c), 0, len(x) - 1))
                    amps.append(max(y[idx], 0.0) * sig0 * np.sqrt(2 * np.pi))
            # else:
            #     centers = list(np.linspace(np.min(x), np.max(x), npeaks))
            #     amps = [max(np.max(y) * sig0 * np.sqrt(2 * np.pi), 1e-6)] * npeaks

            for i in range(npeaks):
                g = GaussianModel(prefix=f"g{i}_")
                model = model + g
                p = g.make_params(center=float(centers[i]), sigma=float(sig0), amplitude=float(amps[i]))
                p[f"g{i}_sigma"].min = 1e-6
                p[f"g{i}_center"].min = float(np.min(x))
                p[f"g{i}_center"].max = float(np.max(x))
                params.update(p)

            # weights = 1 / yerr (safe)
            yerr_arr = np.asarray(yerr)
            if not np.any(yerr_arr > 0):
                yerr_safe = np.ones_like(yerr_arr)
            else:
                pos = yerr_arr[yerr_arr > 0]
                med = float(np.median(pos)) if len(pos) else 1.0
                # med = float(1) if len(pos) else 1.0
                yerr_safe = np.where((yerr_arr > 0) & np.isfinite(yerr_arr), yerr_arr, med)

            result = model.fit(y, params, x=x, weights=1.0 / yerr_safe)

            # overlay
            self.canvas.ax.plot(x, result.best_fit, lw=2, label=f"Fit run {run_id:04d}")
            if show_components:
                comps = result.eval_components(x=x)
                for name, ycomp in comps.items():
                    if name.startswith('g'):
                        self.canvas.ax.plot(x, ycomp, lw=1, alpha=0.8, label=f"{name[:-1]} (run {run_id:04d})")

            self.canvas.refresh(title=self.canvas.ax.get_title(), xlabel=self._xlabel_current)
            self.statusBar().showMessage(f"Fit run {run_id:04d} complete. redχ²={result.redchi:.3g}", 6000)
            # msg = "\n".join(
            #     f"{name}: {par.value:.4g} ± {par.stderr:.2g}" if par.stderr is not None else f"{name}: {par.value:.4g}"
            #     for name, par in result.params.items()
            # )
            # QMessageBox.information(self, "Fit results", msg)
            self._populate_fit_table(result, npeaks, run_id)
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))

    def _populate_fit_table(self, result, npeaks: int, run_id: int):
        """Fill the table with background + Gaussian component parameters and 1σ uncertainties."""
        # rows: 1 background + npeaks gaussians
        rows = 1 + npeaks
        self.fit_table.setRowCount(rows)

        # Background row: merged across all columns
        row = 0
        try:
            b_slope = result.params["bg_slope"]
            b_inter = result.params["bg_intercept"]
            slope_txt = f"{b_slope.value:.4g} ± {b_slope.stderr:.2g}" if b_slope.stderr is not None else f"{b_slope.value:.4g}"
            inter_txt = f"{b_inter.value:.4g} ± {b_inter.stderr:.2g}" if b_inter.stderr is not None else f"{b_inter.value:.4g}"
            txt = f"Background: slope={slope_txt}, intercept={inter_txt}"
        except Exception:
            txt = "Background parameters unavailable"

        self.fit_table.setSpan(row, 0, 1, self.fit_table.columnCount())  # merge full row
        item = QTableWidgetItem(txt)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.fit_table.setItem(row, 0, item)

        # Gaussians
        for i in range(npeaks):
            comp = f"g{i} (run {run_id:04d})"
            amp = result.params.get(f"g{i}_amplitude", None)
            cen = result.params.get(f"g{i}_center", None)
            sig = result.params.get(f"g{i}_sigma", None)

            def fmt(p):
                if p is None:
                    return "—"
                if p.stderr is not None:
                    return f"{p.value:.6g} ± {p.stderr:.2g}"
                return f"{p.value:.6g}"

            amp_txt = fmt(amp)
            cen_txt = fmt(cen)
            sig_txt = fmt(sig)

            # FWHM = 2*sqrt(2*ln2)*sigma
            try:
                fwhm_val = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig.value
                fwhm_err = None if (sig.stderr is None) else 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig.stderr
                if fwhm_err is not None:
                    fwhm_txt = f"{fwhm_val:.6g} ± {fwhm_err:.2g}"
                else:
                    fwhm_txt = f"{fwhm_val:.6g}"
            except Exception:
                fwhm_txt = "—"

            self._set_fit_row(i + 1, comp, amp_txt, cen_txt, sig_txt, fwhm_txt)

    def _set_fit_row(self, row: int, comp: str, amp: str, cen: str, sig: str, fwhm: str):
        items = [comp, amp, cen, sig, fwhm]
        for col, text in enumerate(items):
            it = QTableWidgetItem(text)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.fit_table.setItem(row, col, it)


class FitDialog(QDialog):
    """Dialog to pick run, number of peaks, and options."""
    def __init__(self,parent,run_ids:List[int]):
        super().__init__(parent)
        self.setWindowTitle("Fit peaks")
        self.setModal(True)
        self.run_combo = QComboBox()
        for r in run_ids:
            self.run_combo.addItem(f"{r:04d}",r)

        self.npeaks=QSpinBox()
        self.npeaks.setRange(1,20)
        self.npeaks.setValue(1)
        self.auto_guess = QCheckBox("Auto-guess centers");
        self.auto_guess.setChecked(True)
        self.centers_edit = QLineEdit()
        self.centers_edit.setPlaceholderText("Centers (comma-separated)")
        self.centers_edit.setEnabled(False)
        self.show_components = QCheckBox("Show individual peak components");
        self.show_components.setChecked(True)


        form = QGridLayout(self)
        r = 0
        form.addWidget(QLabel("Run to fit"), r, 0)
        form.addWidget(self.run_combo, r, 1)
        r += 1
        form.addWidget(QLabel("Num. of Gaussians"), r, 0)
        form.addWidget(self.npeaks, r, 1)
        r += 1
        form.addWidget(self.auto_guess, r, 1)
        r += 1
        form.addWidget(QLabel("Centers"), r, 0)
        form.addWidget(self.centers_edit, r, 1)
        r += 1
        form.addWidget(self.show_components, r, 1)
        r += 1

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons, r, 0, 1, 2)

        self.npeaks.valueChanged.connect(self._update_centers_enabled)
        self.auto_guess.toggled.connect(self._update_centers_enabled)
        self._update_centers_enabled()
        self._centers = None

    def _update_centers_enabled(self):
        need_centers = (int(self.npeaks.value()) >=2) or (not self.auto_guess.isChecked())
        self.centers_edit.setEnabled(need_centers)

    def _try_accept(self):
        """Validate centers when required before accepting."""
        need_centers = (int(self.npeaks.value()) >= 2) or (not self.auto_guess.isChecked())
        self._centers = None
        if need_centers:
            txt = self.centers_edit.text().strip()
            if not txt:
                QMessageBox.warning(self, "Centers required","Please enter centers (comma-separated) when fitting ≥2 peaks\n""or when Auto-guess is disabled.")
                return
            try:
                centers = [float(s) for s in txt.split(',') if s.strip()]
            except ValueError:
                QMessageBox.warning(self, "Invalid centers","Centers must be numbers separated by commas (e.g. 1.2, 3.4, 5.6).")
                return
            if len(centers) < int(self.npeaks.value()):
                QMessageBox.warning(self, "Not enough centers",
                                    f"Provided {len(centers)} center(s) but {int(self.npeaks.value())} required.")
                return
            self._centers = centers
        else:
            txt = self.centers_edit.text().strip()
            if txt:
                try:
                    self._centers = [float(s) for s in txt.split(',') if s.strip()]
                except ValueError:
                    self._centers = None
        self.accept()

    def values(self):
        return (
            self.run_combo.currentData(),
            int(self.npeaks.value()),
            bool(self.auto_guess.isChecked()),
            bool(self.show_components.isChecked()),
            self._centers,
        )

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    #test
