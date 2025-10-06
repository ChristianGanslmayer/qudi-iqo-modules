# -*- coding: utf-8 -*-
"""
Structured example GUI (MainWindow + dock widgets) wired to a simple logic
module via Qudi's Connector. The logic centralizes POI parsing/building, and
the GUI delegates to it for consistency with other modules in this repo.

Example config:

gui:
  my_blank_gui:
    module.Class: 'my_blank_gui.my_blank_gui.MyBlankGui'
    connect:
      logic: 'my_blank_logic'
"""

from PySide2 import QtCore, QtWidgets, QtGui

from qudi.core.connector import Connector
from qudi.core.module import GuiBase


# ------ Minimal Dock Widgets -------------------------------------------------

class ControlDockWidget(QtWidgets.QDockWidget):
    sigPingClicked = QtCore.Signal()
    sigGeneratePOIs = QtCore.Signal(list)  # emits list of {'poi': 'POI123'}

    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self._logic = None

        content = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(content)

        # --- Ping row ---
        ping_row = QtWidgets.QHBoxLayout()
        self.ping_button = QtWidgets.QPushButton("Ping logic", content)
        self.reply_label = QtWidgets.QLabel("Reply:", content)
        self.reply_label.setWordWrap(True)
        ping_row.addWidget(self.ping_button)
        ping_row.addStretch(1)
        root.addLayout(ping_row)
        root.addWidget(self.reply_label)

        root.addSpacing(8)
        root.addWidget(self._make_separator())

        # --- POI Builder header ---
        header = QtWidgets.QLabel("<b>POI Builder</b>", content)
        root.addWidget(header)

        # Mode switch
        self.mode_combo = QtWidgets.QComboBox(content)
        self.mode_combo.addItems(["Simple range", "Advanced list"])
        root.addWidget(self.mode_combo)

        # --- Simple range form ---
        simple_box = QtWidgets.QGroupBox("Simple range", content)
        form = QtWidgets.QFormLayout(simple_box)
        self.prefix_edit = QtWidgets.QLineEdit("POI", simple_box)
        self.start_spin = QtWidgets.QSpinBox(simple_box)
        self.start_spin.setRange(0, 10_000_000)
        self.start_spin.setValue(101)
        self.end_spin = QtWidgets.QSpinBox(simple_box)
        self.end_spin.setRange(0, 10_000_000)
        self.end_spin.setValue(153)
        # Optional: zero-padding (disabled by default to preserve current behavior)
        # self.pad_width_spin = QtWidgets.QSpinBox(simple_box)
        # self.pad_width_spin.setRange(0, 12)
        # self.pad_width_spin.setValue(0)
        # form.addRow("Zero-pad width", self.pad_width_spin)

        form.addRow("Prefix", self.prefix_edit)
        form.addRow("Start", self.start_spin)
        form.addRow("End", self.end_spin)

        # --- Advanced input ---
        adv_box = QtWidgets.QGroupBox("Advanced list", content)
        adv_layout = QtWidgets.QVBoxLayout(adv_box)
        self.adv_edit = QtWidgets.QLineEdit("POI101-153, TEST1-3, NV001-010", adv_box)
        self.adv_help = QtWidgets.QLabel(
            "Comma-separated tokens. Each token is either a single ID (e.g. POI150) "
            "or a range with shared prefix (e.g. POI101-153). Zero-padding inferred.",
            adv_box,
        )
        self.adv_help.setWordWrap(True)
        adv_layout.addWidget(self.adv_edit)
        adv_layout.addWidget(self.adv_help)

        # --- Preview + Generate ---
        preview_row = QtWidgets.QHBoxLayout()
        self.preview_label = QtWidgets.QLabel("0 POIs", content)
        self.generate_btn = QtWidgets.QPushButton("Generate POIs", content)
        self.generate_btn.setEnabled(False)
        preview_row.addWidget(self.preview_label, 1)
        preview_row.addWidget(self.generate_btn)

        # Stack the two modes
        self.mode_stack = QtWidgets.QStackedWidget(content)
        self.mode_stack.addWidget(simple_box)  # index 0
        self.mode_stack.addWidget(adv_box)     # index 1

        root.addWidget(self.mode_stack)
        root.addLayout(preview_row)

        content.setLayout(root)
        self.setWidget(content)

        # --- Signals ---
        self.ping_button.clicked.connect(self.sigPingClicked.emit)
        self.mode_combo.currentIndexChanged.connect(self.mode_stack.setCurrentIndex)

        # Recompute preview whenever inputs change
        self.prefix_edit.textChanged.connect(self._update_preview)
        self.start_spin.valueChanged.connect(self._update_preview)
        self.end_spin.valueChanged.connect(self._update_preview)
        # self.pad_width_spin.valueChanged.connect(self._update_preview)
        self.adv_edit.textChanged.connect(self._update_preview)
        self.mode_combo.currentIndexChanged.connect(self._update_preview)
        self.generate_btn.clicked.connect(self._emit_pois)

        # Initial
        self._update_preview()

    # external dependencies --------------------------------------------------
    def set_logic(self, logic):
        """Inject resolved logic instance."""
        self._logic = logic
        self._update_preview()

    # internals --------------------------------------------------------------
    def _make_separator(self):
        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    def _emit_pois(self):
        pois = self._build_pois()
        if pois:
            self.sigGeneratePOIs.emit(pois)

    def _update_preview(self):
        pois = self._build_pois()
        if self._logic is not None:
            text = self._logic.preview_text(pois)
        else:
            # Fallback preview in GUI (kept minimal)
            n = len(pois)
            if n == 0:
                text = "0 POIs"
            elif n <= 6:
                text = f"{n} POIs: " + ", ".join(p['poi'] for p in pois)
            else:
                text = f"{n} POIs: {pois[0]['poi']}, {pois[1]['poi']}, …, {pois[-2]['poi']}, {pois[-1]['poi']}"
        self.preview_label.setText(text)
        self.generate_btn.setEnabled(len(pois) > 0)

    def _build_pois(self):
        if self.mode_stack.currentIndex() == 0:
            # Simple range (no padding to preserve current behavior)
            prefix = self.prefix_edit.text().strip()
            start = int(self.start_spin.value())
            end = int(self.end_spin.value())
            if self._logic is not None:
                return list(self._logic.build_pois_simple(prefix, start, end))
            # GUI fallback
            if not prefix or end < start:
                return []
            return [{'poi': f"{prefix}{i}"} for i in range(start, end + 1)]
        else:
            # Advanced parser (delegate to logic)
            if self._logic is not None:
                return list(self._logic.parse_pois_advanced(self.adv_edit.text()))
            # Minimal GUI fallback (no padding inference)
            s = self.adv_edit.text()
            out = []
            for token in [t.strip() for t in s.split(',') if t.strip()]:
                out.append({'poi': token})
            # Deduplicate while preserving order
            seen = set()
            uniq = []
            for d in out:
                if d['poi'] in seen:
                    continue
                seen.add(d['poi'])
                uniq.append(d)
            return uniq


class OutputDockWidget(QtWidgets.QDockWidget):
    """Output dock: placeholder area to display text/logs."""
    def __init__(self, parent=None):
        super().__init__("Output", parent)
        content = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(content)

        self.text = QtWidgets.QPlainTextEdit(content)
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Output will appear here…")

        layout.addWidget(self.text)
        content.setLayout(layout)
        self.setWidget(content)

    def append_line(self, line: str):
        self.text.appendPlainText(line)


# ------ Main Window ----------------------------------------------------------

class BlankMainWindow(QtWidgets.QMainWindow):
    """Main window (menus + status + actions), holds the dock widgets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("qudi: Blank GUI")

        # Menu bar
        menu_bar = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = menu_bar.addMenu("File")
        self.action_close = QtWidgets.QAction("Close", self)
        self.action_close.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton))
        self.action_close.triggered.connect(self.close)
        file_menu.addAction(self.action_close)

        # View menu
        view_menu = menu_bar.addMenu("View")
        self.action_view_controls = QtWidgets.QAction("Show Controls", self, checkable=True, checked=True)
        self.action_view_output = QtWidgets.QAction("Show Output", self, checkable=True, checked=True)
        self.action_view_default = QtWidgets.QAction("Restore Default", self)

        view_menu.addAction(self.action_view_controls)
        view_menu.addAction(self.action_view_output)
        view_menu.addSeparator()
        view_menu.addAction(self.action_view_default)

        # Status bar
        status_bar = QtWidgets.QStatusBar(self)
        status_bar.setStyleSheet("QStatusBar::item { border: 0px }")
        self.setStatusBar(status_bar)

        status_widget = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(status_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)

        bold = QtGui.QFont()
        bold.setBold(True)
        bold.setPointSize(10)

        label_l = QtWidgets.QLabel("Status:", status_widget)
        label_l.setFont(bold)
        label_l.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status_value = QtWidgets.QLabel("Idle", status_widget)
        self.status_value.setFont(bold)

        grid.addWidget(label_l, 0, 0)
        grid.addWidget(self.status_value, 0, 1)
        status_widget.setLayout(grid)

        status_bar.addPermanentWidget(status_widget, 1)


# ------ GUI Module (Qudi) ---------------------------------------------------

class MyBlankGui(GuiBase):
    """
    Minimal Qudi GUI module wired to MyBlankLogic.

    Example config:

    gui:
      my_blank_gui:
        module.Class: 'my_blank_gui.my_blank_gui.MyBlankGui'
        connect:
          logic: 'my_blank_logic'
    """

    # Connector to your logic module
    logic = Connector(name='logic', interface='MyBlankLogic', optional=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None
        self._controls = None
        self._output = None

    # --- Lifecycle ----------------------------------------------------------

    def on_activate(self):
        """Build UI and connect signals to logic."""
        logic = self.logic()

        # Build main window & docks
        self._mw = BlankMainWindow()
        self._mw.setDockNestingEnabled(True)

        self._controls = ControlDockWidget()
        self._controls.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable | QtWidgets.QDockWidget.DockWidgetMovable
        )
        self._controls.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self._mw.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._controls)

        self._output = OutputDockWidget()
        self._output.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)
        self._output.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._output)

        # View menu actions ↔ dock visibility
        self._controls.visibilityChanged.connect(self._mw.action_view_controls.setChecked)
        self._mw.action_view_controls.toggled.connect(self._controls.setVisible)
        self._output.visibilityChanged.connect(self._mw.action_view_output.setChecked)
        self._mw.action_view_output.toggled.connect(self._output.setVisible)
        self._mw.action_view_default.triggered.connect(self.restore_default_view)

        # Hook up controls and logic
        self._controls.set_logic(logic)
        self._controls.sigPingClicked.connect(self._on_ping_clicked)
        self._controls.sigGeneratePOIs.connect(self._on_generate_pois)

        # Logic → GUI
        logic.sigLog.connect(self._output.append_line, QtCore.Qt.QueuedConnection)
        logic.sigStatus.connect(self._mw.status_value.setText, QtCore.Qt.QueuedConnection)

        # Show default arrangement and window
        self.restore_default_view()
        self.show()

    def on_deactivate(self):
        """Disconnect everything cleanly."""
        try:
            if self._controls is not None:
                self._controls.sigPingClicked.disconnect()
                self._controls.sigGeneratePOIs.disconnect()
                self._controls.visibilityChanged.disconnect()
                self._mw.action_view_controls.toggled.disconnect()
            if self._output is not None:
                self._output.visibilityChanged.disconnect()
                self._mw.action_view_output.toggled.disconnect()
            self._mw.action_view_default.triggered.disconnect()
            # logic signals
            logic = self.logic()
            logic.sigLog.disconnect()
            logic.sigStatus.disconnect()
        except Exception:
            pass

        if self._mw is not None:
            self._mw.close()

        self._mw = None
        self._controls = None
        self._output = None

    # --- Required by Qudi tray ---------------------------------------------

    def show(self):
        """Make window visible and bring to front (tray calls this)."""
        if self._mw is not None:
            self._mw.show()
            self._mw.raise_()
            self._mw.activateWindow()

    # --- Layout helpers -----------------------------------------------------

    def restore_default_view(self):
        """Restore a simple default layout."""
        if self._mw is None:
            return

        if self._controls is not None:
            self._controls.show()
            self._controls.setFloating(False)
        if self._output is not None:
            self._output.show()
            self._output.setFloating(False)

        self._mw.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._controls)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._output)

    # --- Slots / handlers ---------------------------------------------------

    @QtCore.Slot()
    def _on_ping_clicked(self):
        """Call logic.ping() and display the result."""
        try:
            reply = self.logic().ping()
        except Exception as exc:
            reply = f"Error: {exc!r}"

        self._controls.reply_label.setText(f"Reply: {reply}")
        self._mw.status_value.setText("Pinged")
        if self._output is not None:
            self._output.append_line(f"Ping -> {reply}")

    @QtCore.Slot(list)
    def _on_generate_pois(self, pois):
        """Simple demo sink to show generated POIs in the Output dock."""
        if not pois:
            return
        text = self.logic().preview_text(pois)
        self._output.append_line(text)

