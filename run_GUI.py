import sys
import os
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox, QGridLayout,
    QHBoxLayout, QVBoxLayout, QStatusBar, QColorDialog
)

# Matplotlib Qt canvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

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
            nums = int(nums)
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
        self.setMinimumSize(1100, 680)

        # Widgets
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose parent folder that contains /expXXXX/Datafiles …")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.on_browse)

        self.instrument_edit = QLineEdit()
        self.instrument_edit.setPlaceholderText("e.g., HB1A, CTAX, TRIAX...")

        self.exp_spin = QSpinBox()
        self.exp_spin.setRange(0, 999999)
        self.exp_spin.setValue(1)
        self.exp_spin.setPrefix("exp ")

        # Multi-run entry
        self.run_edit = QLineEdit()
        self.run_edit.setPlaceholderText("Run(s): e.g. 12,14,18-21,(7,15,2)")

        self.temp_label_edit = QLineEdit()
        self.temp_label_edit.setPlaceholderText("Temperature column header in file (e.g., T_sample)")

        self.colors_edit = QLineEdit()
        self.colors_edit.setPlaceholderText("Colors (optional): e.g. C0,#1f77b4,red,#00aa55,black")
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
        top_layout.addLayout(form)
        top_layout.addWidget(self.toolbar)
        top_layout.addWidget(self.canvas, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.save_fig_btn)
        top_layout.addLayout(button_row)

        self.setCentralWidget(top)
        self.setStatusBar(QStatusBar())

        # State
        self._last_loaded = None  # type: Optional[dict]

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
                raise ValueError("Please enter at least one run number (e.g. '12,14,18-21').")

            colors = parse_colors(color_spec, len(runs))
            if self.clear_check.isChecked():
                self.canvas.ax.clear()

            self.statusBar().showMessage("Loading…", 3000)
            loader = import_triX_single(instrument=instrument, exp=exp, label_T=temp_label)
            last_label_for_title = None

            for i, r in enumerate(runs):
                label, x, y, yerr = loader.load_data(folder, r, nor_to_cps=norm, name_x=x_override)
                run_lbl = f"run {r:04d}"
                self.canvas.plot_xy(x, y, yerr=yerr, label='{} - {} K'.format(run_lbl,label['temperature']), color=colors[i])
                last_label_for_title = label

            title = ""
            if last_label_for_title:
                # title = f"{last_label_for_title.get('samplename','')}  T={last_label_for_title.get('temperature','?')}±{last_label_for_title.get('tem_error','?')} K"
                title = f"{last_label_for_title.get('samplename', '')} Run(s) {run_spec}"
            self.canvas.refresh(title=title,xlabel=label['x'])

            self._last_loaded = {
                'instrument': instrument, 'exp': exp, 'runs': runs,
                'norm': norm, 'folder': folder, 'label': last_label_for_title,
                'colors': colors
            }
            self.statusBar().showMessage(
                f"Loaded runs: {', '.join(f'{r:04d}' for r in runs)} from exp{exp:04d}", 5000)
            self.save_fig_btn.setEnabled(True)
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


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
