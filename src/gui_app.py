"""
PyQt5 GUI for the spiral segmentation pipeline.

Folder / queue workflow:
  1. "Open Folder..." expects the assumed layout pXXX/spiral/*.png -- you
     select the pXXX (patient) folder itself, and it looks inside that
     folder's "spiral" subfolder for the actual .png images.
  2. The .pkl is saved directly into the pXXX folder you selected,
     alongside "spiral" rather than inside it, named after that folder.
     The sidebar shows exactly where it'll land.
  3. A reorder dialog pops up showing the found files; drag them (or use
     Move Up / Move Down) into the order you want them saved in, then
     confirm. That order becomes both the processing queue and the
     "import_order" written into the final .pkl.
  4. The main window shows a sidebar list of every file in the queue, with
     a status indicator (pending / current / done). Completed images can
     be clicked to see the saved result at any time, in any order
     (read-only -- see "Edit" below to adjust one). Pending images
     that aren't up yet can also be clicked, but only show a read-only
     raw-image preview (no segmentation is run) -- segmentation itself
     still has to happen strictly in order.
  5. If a previous session's checkpoint is found (in the parent folder),
     you're asked whether to resume (keeping the prior order + completed
     work) or start over.

Per-image workflow:
  1. Load image, compute skeleton + segments (core.prep_mask_and_segments).
     Every segment starts as REMOVE (default/unmarked, unhighlighted --
     excluded from both the spiral and template reconstructions until
     you explicitly mark it).
  2. Classify segments:
       - Click mode: left-click a segment to advance it through
         REMOVE (unmarked) -> SPIRAL (green) -> TEMPLATE (red) ->
         BOTH (blue) -> back to REMOVE. Right-click steps backward
         through the same cycle, so you can nudge past an overshoot
         either direction.
       - Box-select mode: drag a rectangle on the right panel, then use
         the "Mark selected as..." buttons to bulk-label everything the
         box touched (Spiral / Template / Remove / Both).
       - Freehand marker mode: draw a line (click-drag-release) across
         any segments you want to select -- anything the drawn line
         passes near gets selected, then use the same "Mark selected
         as..." buttons to label them. Handy for a wobbly cluster of
         segments a rectangle wouldn't cleanly capture.
       - Use the toolbar above the canvas to zoom to a rectangle or pan
         for a closer look before clicking/dragging -- click the active
         toolbar tool again to turn it off and go back to normal
         correction, since they share the same mouse drag. The zoom
         level stays put across clicks/selections on the same image (it
         only resets when you move to different content, e.g. the next
         image, or when you reset the view yourself via the toolbar's
         home button).
  3. "Manual classify (fallback)" opens a one-by-one dialog that walks
     through every segment in turn, for when you'd rather not click on
     the image directly.
  4. "Split segment" (rare to use), under the right (segmented) panel,
     opens a small popup -- with its own zoom/pan toolbar for a precise
     click -- where you click the exact point where two curves cross or
     touch but weren't split there automatically. Most commonly this is a
     *tangential* contact, where two curves just graze each other rather
     than crossing at a clear X; that case often has no distinguishing
     pixel pattern for the automatic branch-point detector to find, so if
     you can see it's a crossing and the segment spans across it anyway,
     split it here manually. Splits apply live as you click so you can
     watch each one, but nothing is kept unless you hit Done -- Cancel
     discards every split made in that dialog session. Both resulting
     pieces keep whatever label the original segment had; switch back to
     Click mode afterward to classify them independently.
  5. "Accept segmentation" builds the spiral/template images, then prompts
     you to click the spiral center and template center -- the click
     snaps to the nearest point actually on that curve (so you don't have
     to be pixel-perfect), and the picked point is marked with a target
     reticle.
  6. "Save & Next" checkpoints to disk and advances to the next image.

Run with:  python gui_app.py
"""

import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector, RectangleSelector

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

import core


LABEL_COLORS = {
    core.TEMPLATE: "red",
    core.SPIRAL: "green",
    core.BOTH: "blue",
    # core.REMOVE is intentionally absent -- it's the default/unmarked
    # state and shown unhighlighted (excluded from both reconstructions).
}
DEFAULT_OUTPUT_SIZE = (300, 300)

# Shared look for small secondary-action buttons (Save and Close, Split
# segment) so they read as a consistent, compact, outlined style.
COMPACT_BUTTON_STYLE = (
    "font-size: 11px; padding: 3px 10px; border-radius: 8px; "
    "background: transparent; border: 1px solid #888; color: #333;"
)
COMPACT_BUTTON_HEIGHT = 24

STATUS_PENDING, STATUS_CURRENT, STATUS_DONE = "pending", "current", "done"
STATUS_ICON = {STATUS_PENDING: "\u23f3", STATUS_CURRENT: "\u25b6", STATUS_DONE: "\u2705"}


# ==========================================================================
# Select dialog -- shown right after picking a folder, before reordering
# ==========================================================================

class SelectFilesDialog(QDialog):
    """Lets the user check/uncheck which of the found .png files to
    actually include in the processing queue."""

    def __init__(self, parent, filenames):
        super().__init__(parent)
        self.setWindowTitle("Select images to include")
        self.resize(420, 500)

        layout = QVBoxLayout(self)
        select_help_label = QLabel(
            "Check the images you want to process. Unchecked images are "
            "skipped entirely."
        )
        select_help_label.setWordWrap(True)
        layout.addWidget(select_help_label)

        self.list_widget = QListWidget()
        for name in filenames:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_all = QPushButton("Select All")
        self.btn_none = QPushButton("Select None")
        self.btn_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        self.btn_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        layout.addLayout(btn_row)

        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.list_widget.itemChanged.connect(self._update_count)
        self._update_count()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Continue")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def _update_count(self, *_):
        n = sum(
            1 for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        )
        self.count_label.setText(f"{n} of {self.list_widget.count()} selected")

    def _on_accept(self):
        if not self.selected_filenames():
            QMessageBox.warning(self, "Nothing selected",
                                 "Select at least one image to continue.")
            return
        self.accept()

    def selected_filenames(self):
        return [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        ]


# ==========================================================================
# Reorder dialog -- shown right after picking a folder. Also where the
# per-image "index name" (sidebar/saved-file label) gets set, defaulting
# to everything before the first '_' (or the whole filename if there's no
# '_'), editable per row.
# ==========================================================================

class ReorderDialog(QDialog):
    """Lets the user drag-reorder (or use Move Up/Down) the list of .png
    files found in the chosen folder before processing begins. That order
    becomes the processing queue. Just filenames here -- naming (the
    short "index name" used in the sidebar and saved .pkl) is handled
    separately, automatically at import time and editable afterward from
    the sidebar, not here."""

    def __init__(self, parent, filenames):
        super().__init__(parent)
        self.setWindowTitle("Order images for processing")
        self.resize(420, 500)

        layout = QVBoxLayout(self)
        reorder_help_label = QLabel(
            "Drag to reorder, or use the buttons below. This order is used "
            "both for processing and for how images are saved in the "
            "final file."
        )
        reorder_help_label.setWordWrap(True)
        layout.addWidget(reorder_help_label)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.addItems(filenames)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_up = QPushButton("Move Up")
        self.btn_down = QPushButton("Move Down")
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)
        btn_row.addWidget(self.btn_up)
        btn_row.addWidget(self.btn_down)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Confirm Order && Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.list_widget.currentRow()
        if 0 <= row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)

    def ordered_filenames(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]


# ==========================================================================
# Manual one-by-one fallback dialog (equivalent of the original
# "classify each point" method)
# ==========================================================================

class ManualClassifyDialog(QDialog):
    def __init__(self, parent, segments, labels, mask_skel, crossing_points_thick):
        super().__init__(parent)
        self.setWindowTitle("Manual classify (fallback)")
        self.resize(700, 700)
        self.segments = segments
        self.labels = labels  # shared reference -- edits apply directly
        self.mask_skel = mask_skel
        self.crossing_points_thick = crossing_points_thick
        self.idx = 0

        layout = QVBoxLayout(self)
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)

        btn_row = QHBoxLayout()
        self.btn_spiral = QPushButton("Spiral (1)")
        self.btn_template = QPushButton("Template (2)")
        self.btn_both = QPushButton("Both (3)")
        self.btn_skip = QPushButton("Skip / Neither (Enter)")
        self.btn_undo = QPushButton("Undo last")
        for b in (self.btn_spiral, self.btn_template, self.btn_both,
                  self.btn_skip, self.btn_undo):
            btn_row.addWidget(b)
            # Prevent Qt's default-button mechanism from swallowing Enter
            # before our keyPressEvent handles it.
            b.setAutoDefault(False)
            b.setDefault(False)
        layout.addLayout(btn_row)

        self.btn_spiral.clicked.connect(lambda: self._classify(core.SPIRAL))
        self.btn_template.clicked.connect(lambda: self._classify(core.TEMPLATE))
        self.btn_both.clicked.connect(lambda: self._classify(core.BOTH))
        self.btn_skip.clicked.connect(lambda: self._classify(core.REMOVE))
        self.btn_undo.clicked.connect(self._undo)

        self.btn_done = QPushButton("Done")
        self.btn_done.setAutoDefault(False)
        self.btn_done.setDefault(False)
        self.btn_done.clicked.connect(self.accept)
        layout.addWidget(self.btn_done)

        self.history = []
        self._redraw()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_1:
            self._classify(core.SPIRAL)
        elif key == Qt.Key_2:
            self._classify(core.TEMPLATE)
        elif key == Qt.Key_3:
            self._classify(core.BOTH)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._classify(core.REMOVE)
        else:
            super().keyPressEvent(event)

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.mask_skel & ~self.crossing_points_thick, cmap="gray")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Show already-classified segments in the same colors as the main
        # click-to-toggle view (red=template, green=spiral, blue=both).
        for i, coords in enumerate(self.segments):
            color = LABEL_COLORS.get(self.labels[i])
            if color:
                self.ax.plot(coords[:, 1], coords[:, 0], ".", color=color, markersize=2)

        if self.idx < len(self.segments):
            coords = self.segments[self.idx]
            self.ax.plot(coords[:, 1], coords[:, 0], "y.", markersize=6)
            self.status_label.setText(
                f"Segment {self.idx + 1} / {len(self.segments)}   "
                f"(keys: 1=spiral, 2=template, 3=both, Enter=skip)"
            )
        else:
            self.status_label.setText("All segments classified.")
        self.canvas.draw_idle()

    def _classify(self, label):
        if self.idx >= len(self.segments):
            return
        self.history.append(self.idx)
        self.labels[self.idx] = label
        self.idx += 1
        self._redraw()

    def _undo(self):
        if not self.history:
            return
        self.idx = self.history.pop()
        self._redraw()


# ==========================================================================
# Split-segment dialog -- rarely-used manual override for crossings the
# automatic branch-point detector missed (most often a tangential contact)
# ==========================================================================

class SplitSegmentDialog(QDialog):
    """Popup for the manual split action. Click the exact point where you
    can see two curves cross/touch but weren't split there automatically.
    Delegates the actual split logic to the parent MainWindow's
    _split_segment_at, so this dialog is just an alternate place to click
    -- everything it does is reflected live in the main window too.

    Splits happen live as you click (so you can watch each one), but
    nothing is permanent until you hit Done -- MainWindow snapshots the
    segments/labels before opening this dialog and restores that snapshot
    if you hit Cancel instead."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Split segment -- click the crossing point")
        self.resize(600, 680)
        self.last_click_point = None  # (row, col) -- highlighted yellow

        layout = QVBoxLayout(self)
        split_help_label = QLabel(
            "Click the exact point where two curves cross or touch but "
            "weren't split automatically (e.g. a tangential contact). "
            "You can click more than once if needed. Use the toolbar to "
            "zoom in first for a precise click -- turn the zoom tool back "
            "off (click its icon again) before clicking to split."
        )
        split_help_label.setWordWrap(True)
        layout.addWidget(split_help_label)

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._preserve_view = False

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_done = QPushButton("Done")
        self.btn_done.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_done)
        layout.addLayout(btn_row)

        self._redraw()

    def _redraw(self):
        preserve = self._preserve_view
        self._preserve_view = False
        if preserve:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        self.ax.clear()
        mw = self.main_window
        base = mw.mask_skel & ~mw.crossing_points_thick
        self.ax.imshow(base, cmap="gray")
        for i, coords in enumerate(mw.segments):
            color = LABEL_COLORS.get(mw.labels[i])
            if color:
                self.ax.plot(coords[:, 1], coords[:, 0], ".",
                              color=color, markersize=2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if self.last_click_point is not None:
            row, col = self.last_click_point
            self.ax.plot(col, row, marker="o", markersize=10,
                          markerfacecolor="yellow", markeredgecolor="black",
                          markeredgewidth=1)

        if preserve:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        row, col = event.ydata, event.xdata
        self.last_click_point = (row, col)
        before = len(self.main_window.segments)
        self.main_window._split_segment_at(row, col)
        after = len(self.main_window.segments)
        if after > before:
            self.status_label.setText(
                f"Split done. Total segments now: {after}. Keep clicking "
                f"if there's another crossing to fix, or hit Done."
            )
        else:
            self.status_label.setText(
                "Nothing to split there -- click closer to the actual "
                "crossing/contact point."
            )
        self._preserve_view = True
        self._redraw()


# ==========================================================================
# Main window
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spiral Segmentation")
        self.resize(1200, 850)

        self.folder = None        # the folder of .png images (where they're read from)
        self.save_dir = None      # one level up from folder -- where the .pkl is written
        self.save_tag = None      # save_dir's basename, used in the .pkl filename
        self.filenames = []       # ordered list of basenames (no extension)
        self.statuses = []        # parallel list: STATUS_PENDING/CURRENT/DONE
        self.state = None         # accumulated pickle-able results dict
        self.work_pos = -1
        self.output_size = DEFAULT_OUTPUT_SIZE

        # Cache of (img, mask_skel, crossing_points, crossing_points_thick,
        # segments) keyed by queue position. Populated the first time an
        # image is processed; "Edit" reuses this instead of re-running
        # the image-processing pipeline.
        self._segment_cache = {}

        # Set to a queue position while an "Edit" session (entered from
        # review) is in progress -- lets Cancel Edit know which saved
        # review state to revert back to if you bail without saving.
        self._editing_from_review_pos = None

        # Per-image state
        self.img = None
        self.mask_skel = None
        self.crossing_points = None
        self.crossing_points_thick = None
        self.segments = []
        self.labels = None
        self.box_selection = []
        self.mode = "click"  # "click", "box", or "freehand"
        self.awaiting_center = None
        self.center_spiral = None
        self.center_template = None
        self.final_spiral = None
        self.final_template = None
        self.stage = "segment"  # "segment" -> "centers"

        # When True, the NEXT _redraw() call preserves the current zoom/pan
        # (xlim/ylim) instead of resetting to full view -- set this before
        # calling _redraw() from an interaction that updates the SAME image
        # (a label toggle, a selection, a center pick), not when loading
        # genuinely different content (a new image, or the centers stage's
        # differently-sized arrays).
        self._preserve_view = False

        self._build_ui()

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # --- Sidebar ---
        sidebar = QVBoxLayout()
        self.btn_open = QPushButton("Open Folder...")
        self.btn_open.clicked.connect(self.choose_folder)
        sidebar.addWidget(self.btn_open)
        sidebar.addWidget(QLabel("Queue: (double-click a name to rename it)"))
        self.queue_list = QListWidget()
        self.queue_list.itemClicked.connect(self._on_queue_item_clicked)
        self.queue_list.itemChanged.connect(self._on_queue_item_renamed)
        self._suppress_rename_handler = False
        sidebar.addWidget(self.queue_list, stretch=1)

        self.save_path_label = QLabel("Saving to: (no folder selected)")
        self.save_path_label.setWordWrap(True)
        self.save_path_label.setStyleSheet("color: #666; font-size: 11px;")
        sidebar.addWidget(self.save_path_label)

        self.btn_save_close = QPushButton("Save and Close")
        self.btn_save_close.clicked.connect(self.close)
        self.btn_save_close.setStyleSheet(COMPACT_BUTTON_STYLE)
        self.btn_save_close.setMaximumHeight(COMPACT_BUTTON_HEIGHT)
        sidebar.addWidget(self.btn_save_close)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(260)
        root_layout.addWidget(sidebar_widget)

        # --- Main content ---
        outer = QVBoxLayout()
        root_layout.addLayout(outer, stretch=1)

        self.status_label = QLabel("No folder selected.")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.fig = Figure(figsize=(10, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_left = self.fig.add_subplot(121)
        self.ax_right = self.fig.add_subplot(122)
        outer.addWidget(self.toolbar)
        outer.addWidget(self.canvas, stretch=1)

        split_row = QHBoxLayout()
        self.btn_split = QPushButton("Split segment")
        self.btn_split.setStyleSheet(COMPACT_BUTTON_STYLE)
        self.btn_split.setMaximumHeight(COMPACT_BUTTON_HEIGHT)
        self.btn_split.clicked.connect(self.open_split_dialog)
        self.btn_split.setEnabled(False)
        # Roughly centered under the RIGHT panel (the segmented view) --
        # the two subplots split the canvas about evenly, so biasing the
        # stretch this way lands the button under the right half.
        split_row.addStretch(3)
        split_row.addWidget(self.btn_split)
        split_row.addStretch(1)
        outer.addLayout(split_row)

        toolbar_help_label = QLabel(
            "Use the toolbar's magnifying-glass icon to zoom to a rectangle, "
            "the hand icon to pan, and the home icon to reset the view. "
            "Click the active tool again to turn it off before clicking/"
            "dragging to correct segments below."
        )
        toolbar_help_label.setWordWrap(True)
        outer.addWidget(toolbar_help_label)

        self.rect_selector = RectangleSelector(
            self.ax_right,
            self._on_box_select,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=False,
        )
        self.rect_selector.set_active(False)

        self.lasso_selector = LassoSelector(
            self.ax_right,
            self._on_lasso_select,
            useblit=True,
            button=[1],
        )
        self.lasso_selector.set_active(False)

        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        corr_box = QGroupBox()
        corr_outer = QVBoxLayout(corr_box)
        mode_row = QHBoxLayout()
        self.radio_click = QRadioButton("Click to toggle")
        self.radio_box = QRadioButton("Drag box to bulk-select")
        self.radio_freehand = QRadioButton("Freehand marker (draw a line)")
        self.radio_click.setChecked(True)
        self.radio_click.toggled.connect(self._on_mode_toggle)
        self.radio_box.toggled.connect(self._on_mode_toggle)
        self.radio_freehand.toggled.connect(self._on_mode_toggle)
        mode_row.addWidget(self.radio_click)
        mode_row.addWidget(self.radio_box)
        mode_row.addWidget(self.radio_freehand)
        corr_outer.addLayout(mode_row)

        mark_row = QHBoxLayout()
        self.btn_mark_spiral = QPushButton("Mark selected: Spiral")
        self.btn_mark_template = QPushButton("Mark selected: Template")
        self.btn_mark_remove = QPushButton("Mark selected: Remove")
        self.btn_mark_both = QPushButton("Mark selected: Both")
        for b, lab in (
            (self.btn_mark_spiral, core.SPIRAL),
            (self.btn_mark_template, core.TEMPLATE),
            (self.btn_mark_remove, core.REMOVE),
            (self.btn_mark_both, core.BOTH),
        ):
            b.clicked.connect(lambda _, l=lab: self._apply_bulk_label(l))
            b.setEnabled(False)
            mark_row.addWidget(b)
        corr_outer.addLayout(mark_row)
        outer.addWidget(corr_box)

        # Two rows instead of one long one -- a single unwrapped row of six
        # buttons this wide was forcing the window's minimum width past
        # 2000px, which is why it could shrink freely in height but not
        # width on a real screen.
        action_row1 = QHBoxLayout()
        self.btn_manual = QPushButton("Manual classify (fallback)")
        self.btn_manual.clicked.connect(self.open_manual_dialog)
        action_row1.addWidget(self.btn_manual)

        self.btn_accept = QPushButton("Accept segmentation")
        self.btn_accept.clicked.connect(self.accept_segmentation)
        action_row1.addWidget(self.btn_accept)

        self.btn_pick_spiral_center = QPushButton("Pick spiral center")
        self.btn_pick_spiral_center.clicked.connect(
            lambda: self._start_center_pick("spiral")
        )
        self.btn_pick_spiral_center.setEnabled(False)
        action_row1.addWidget(self.btn_pick_spiral_center)
        outer.addLayout(action_row1)

        action_row2 = QHBoxLayout()
        self.btn_pick_template_center = QPushButton("Pick template center")
        self.btn_pick_template_center.clicked.connect(
            lambda: self._start_center_pick("template")
        )
        self.btn_pick_template_center.setEnabled(False)
        action_row2.addWidget(self.btn_pick_template_center)

        self.btn_save_next = QPushButton("Save && Next")
        self.btn_save_next.clicked.connect(self.save_and_next)
        self.btn_save_next.setEnabled(False)
        action_row2.addWidget(self.btn_save_next)

        self.btn_edit_review = QPushButton("Edit")
        self.btn_edit_review.clicked.connect(self.edit_from_review)
        self.btn_edit_review.setEnabled(False)
        action_row2.addWidget(self.btn_edit_review)
        outer.addLayout(action_row2)

        # Small, out-of-the-way -- only ever relevant while an Edit
        # session (entered from review) is actually in progress.
        cancel_edit_row = QHBoxLayout()
        self.btn_cancel_edit = QPushButton("Cancel Edit")
        self.btn_cancel_edit.setStyleSheet(COMPACT_BUTTON_STYLE)
        self.btn_cancel_edit.setMaximumHeight(COMPACT_BUTTON_HEIGHT)
        self.btn_cancel_edit.clicked.connect(self.cancel_edit_from_review)
        self.btn_cancel_edit.setEnabled(False)
        cancel_edit_row.addStretch(1)
        cancel_edit_row.addWidget(self.btn_cancel_edit)
        outer.addLayout(cancel_edit_row)

        self._set_controls_enabled(False)

    def _set_controls_enabled(self, enabled):
        for w in (
            self.radio_click,
            self.radio_box,
            self.radio_freehand,
            self.btn_split,
            self.btn_manual,
            self.btn_accept,
        ):
            w.setEnabled(enabled)

    # ------------------------------------------------------------ Folder --

    def choose_folder(self):
        parent_folder = QFileDialog.getExistingDirectory(
            self, "Select patient folder (e.g. pXXX -- containing a 'spiral' subfolder)"
        )
        if not parent_folder:
            return

        # Assumed layout: pXXX/spiral/*.png -- you select pXXX itself, and
        # this looks inside its "spiral" subfolder for the actual images.
        folder = os.path.join(parent_folder, "spiral")
        if not os.path.isdir(folder):
            QMessageBox.warning(
                self, "No 'spiral' subfolder found",
                f"Expected to find a 'spiral' subfolder inside:\n{parent_folder}\n\n"
                f"but there isn't one there. Select the patient folder that "
                f"CONTAINS a 'spiral' subfolder with your .png images in it, "
                f"not the images folder itself."
            )
            return

        pngs = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(".png")
        )
        if not pngs:
            QMessageBox.warning(self, "No images found",
                                 f"No .png files were found in {folder}.")
            return
        basenames = [os.path.splitext(f)[0] for f in pngs]

        # The .pkl is saved directly into the folder you selected (pXXX),
        # alongside "spiral" rather than inside it.
        save_dir = parent_folder
        save_tag = os.path.basename(os.path.normpath(save_dir)) or "session"
        tmp_path = os.path.join(save_dir, f"{save_tag}_spiral_tmp.pkl")
        final_path = os.path.join(save_dir, f"{save_tag}_spiral.pkl")

        resumed = False

        if os.path.exists(final_path):
            # Fully completed before -- load it so completed images show
            # up in the sidebar and can be reviewed / redone via "Start
            # Over" (which re-saves straight back into this same file),
            # rather than silently starting a brand new blank session.
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Fully preprocessed file found")
            msg_box.setText(
                "Fully preprocessed file found. View/Edit segmentations, "
                "or start over?"
            )
            view_edit_btn = msg_box.addButton("View/Edit", QMessageBox.YesRole)
            start_over_btn = msg_box.addButton("Start Over", QMessageBox.NoRole)
            msg_box.setDefaultButton(view_edit_btn)
            msg_box.exec_()
            if msg_box.clickedButton() == view_edit_btn:
                with open(final_path, "rb") as f:
                    self.state = pickle.load(f)
                self.filenames = self.state["import_order"]
                # Backward compatibility: older .pkl files (or ones
                # converted from .mat before this existed) won't have
                # these keys -- fall back to identity (index name IS the
                # real basename) so nothing crashes on an older file.
                self.state.setdefault(
                    "basename_by_index",
                    {name: name for name in self.filenames},
                )
                self.state.setdefault("source_filenames", {})
                resumed = True

        elif os.path.exists(tmp_path):
            choice = QMessageBox.question(
                self, "Temporary file found",
                "Temporary file found, resume?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if choice == QMessageBox.Yes:
                with open(tmp_path, "rb") as f:
                    self.state = pickle.load(f)
                self.filenames = self.state["import_order"]
                self.state.setdefault(
                    "basename_by_index",
                    {name: name for name in self.filenames},
                )
                self.state.setdefault("source_filenames", {})
                resumed = True
            else:
                os.remove(tmp_path)

        if not resumed:
            select_dlg = SelectFilesDialog(self, basenames)
            if select_dlg.exec_() != QDialog.Accepted:
                return
            chosen = select_dlg.selected_filenames()

            reorder_dlg = ReorderDialog(self, chosen)
            if reorder_dlg.exec_() != QDialog.Accepted:
                return
            ordered_basenames = reorder_dlg.ordered_filenames()

            # Default index names applied automatically at import time
            # (everything before the first '_', or the whole filename if
            # there's no '_', with a collision-safe fallback) -- rename
            # individual ones later from the sidebar if needed, not here.
            index_names, basename_by_index = core.build_index_name_map(ordered_basenames)

            self.filenames = index_names
            self.state = {
                "import_order": self.filenames,
                "basename_by_index": basename_by_index,
                # Permanent, position-keyed record of the exact real .png
                # each saved image came from -- populated in save_and_next
                # as each image is actually saved (basename_by_index can
                # be renamed later; this can't, so it stays a reliable
                # audit trail even if an index name changes afterward).
                "source_filenames": {},
                "spiral_ims": {}, "spirals": {}, "crossing_points": {},
                "template": {}, "center_template": {}, "center_spiral": {},
            }

        self.folder = folder
        self.save_dir = save_dir
        self.save_tag = save_tag
        self.save_path_label.setText(f"Saving to: {final_path}")
        n = len(self.filenames)
        self.statuses = [
            STATUS_DONE if (i + 1) in self.state["spirals"] else STATUS_PENDING
            for i in range(n)
        ]
        self._rebuild_queue_list()

        self.work_pos = -1
        for i, s in enumerate(self.statuses):
            if s != STATUS_DONE:
                self.work_pos = i - 1
                break
        else:
            # Everything's already done -- e.g. you picked "View/Edit" on a
            # fully preprocessed folder. There's no "next" image to load,
            # so don't try (that's what was popping the "No more images in
            # the queue" dialog immediately on open). Just populate the
            # sidebar and let the person click into whichever one they
            # want to look at.
            self.work_pos = -1
            self.stage = "segment"
            self.status_label.setText(
                "All images in this folder are already complete. Click "
                "any item in the sidebar to view or redo it."
            )
            return

        self._load_at(self.work_pos + 1)

    def _rebuild_queue_list(self):
        self._suppress_rename_handler = True
        self.queue_list.clear()
        for name, status in zip(self.filenames, self.statuses):
            item = QListWidgetItem(f"{STATUS_ICON[status]}  {name}")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.queue_list.addItem(item)
        self._suppress_rename_handler = False

    def _persist_checkpoint(self):
        """Write current self.state to whichever checkpoint file is
        appropriate (final .pkl if everything's done, otherwise the tmp
        one) -- used after an in-place rename so it isn't silently lost
        if you close without doing anything else afterward."""
        if self.save_dir is None or self.save_tag is None:
            return
        if self.statuses and all(s == STATUS_DONE for s in self.statuses):
            path = os.path.join(self.save_dir, f"{self.save_tag}_spiral.pkl")
        else:
            path = os.path.join(self.save_dir, f"{self.save_tag}_spiral_tmp.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.state, f)

    def _resolve_real_basename(self, fname):
        """Look up the real on-disk basename for an index name, self-
        healing a stale/wrong basename_by_index entry (most commonly a
        .pkl converted from .mat without --images-dir, so the mapping
        fell back to an identity guess that doesn't match the real
        filenames) by checking what's actually in the folder. If a
        correction is found, it's saved back into basename_by_index and
        persisted, so this only ever needs to self-heal once per image."""
        stored = self.state.get("basename_by_index", {}).get(fname, fname)
        resolved = core.resolve_real_basename(self.folder, fname, fallback_basename=stored)
        if resolved != stored:
            self.state.setdefault("basename_by_index", {})[fname] = resolved
            self._persist_checkpoint()
        return resolved

    def _safe_get_or_compute_segments(self, pos):
        """Wraps _get_or_compute_segments so a naming mismatch it can't
        resolve shows a clear, actionable dialog -- listing what's
        actually in the folder -- instead of crashing the whole app with
        an unhandled FileNotFoundError. The self-healing resolver covers
        the common naming patterns, but nothing can guess an arbitrary
        one, so this is the backstop for whatever it can't figure out.
        Returns None on failure; callers should check for that and bail
        out of whatever they were doing without changing any state."""
        try:
            return self._get_or_compute_segments(pos)
        except FileNotFoundError:
            fname = self.filenames[pos]
            real_basename = self.state.get("basename_by_index", {}).get(fname, fname)
            try:
                available = sorted(
                    f for f in os.listdir(self.folder) if f.lower().endswith(".png")
                )
            except OSError:
                available = []
            listing = "\n".join(available[:30]) + ("\n..." if len(available) > 30 else "")
            QMessageBox.warning(
                self, "Image file not found",
                f"Couldn't find an image file for '{fname}' "
                f"(tried '{real_basename}.png') in:\n{self.folder}\n\n"
                f"Files actually in that folder:\n{listing or '(none found)'}\n\n"
                f"You can rename '{fname}' from the sidebar to match the "
                f"correct file, then try again."
            )
            return None

    def _on_queue_item_renamed(self, item):
        """Rename an image's short index name right from the sidebar --
        purely a label change. The per-image data (spirals[ii],
        centers[ii], etc.) is keyed by queue POSITION, never by name, so
        renaming can never disturb already-saved segmentation data; it
        only needs to keep self.filenames, state['import_order'], and
        state['basename_by_index'] (which still points at the same real
        file on disk) in sync."""
        if self._suppress_rename_handler:
            return
        row = self.queue_list.row(item)
        if row < 0 or row >= len(self.filenames):
            return

        status = self.statuses[row]
        prefix = f"{STATUS_ICON[status]}  "
        raw_text = item.text()
        new_name = raw_text[len(prefix):] if raw_text.startswith(prefix) else raw_text
        new_name = new_name.replace(" ", "").strip()
        old_name = self.filenames[row]

        def revert():
            self._suppress_rename_handler = True
            item.setText(f"{prefix}{old_name}")
            self._suppress_rename_handler = False

        if not new_name:
            QMessageBox.warning(self, "Invalid name", "Index name can't be empty.")
            revert()
            return
        if new_name != old_name and new_name in self.filenames:
            QMessageBox.warning(self, "Duplicate name",
                                 f"'{new_name}' is already used by another image.")
            revert()
            return
        if new_name == old_name:
            revert()  # just re-normalize the displayed text
            return

        self.filenames[row] = new_name
        self.state["import_order"][row] = new_name
        basename_by_index = self.state.setdefault("basename_by_index", {})
        real_basename = basename_by_index.pop(old_name, old_name)
        basename_by_index[new_name] = real_basename

        if row == self.work_pos:
            self.current_fname = new_name

        self._suppress_rename_handler = True
        item.setText(f"{prefix}{new_name}")
        self._suppress_rename_handler = False

        self._persist_checkpoint()

    def _next_required_pos(self):
        """Smallest index that isn't DONE yet -- the only position you're
        allowed to start new work on. Returns None if everything's done."""
        for i, s in enumerate(self.statuses):
            if s != STATUS_DONE:
                return i
        return None

    def _on_queue_item_clicked(self, item):
        row = self.queue_list.row(item)
        if row == self.work_pos:
            return

        if self.statuses[row] == STATUS_DONE:
            # Previewing a completed image is always allowed, in any order.
            if self.stage not in ("segment", "review", "preview") and not self._confirm_discard():
                return
            self._load_review(row)
            return

        # Not yet done -- only the next required position in sequence can
        # be *edited*. Anything later can still be looked at (raw image
        # only, read-only) via preview, just not segmented out of turn.
        next_required = self._next_required_pos()
        if next_required is not None and row != next_required:
            if self.stage not in ("segment", "review", "preview") and not self._confirm_discard():
                return
            self._load_preview(row)
            return

        if self.stage not in ("segment", "review", "preview") and not self._confirm_discard():
            return
        self._load_at(row)

    def _confirm_discard(self):
        choice = QMessageBox.question(
            self, "Discard unsaved progress?",
            "This image's segmentation hasn't been saved yet. Discard it "
            "and navigate away?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return choice == QMessageBox.Yes

    # ------------------------------------------------------------- Image --

    def _get_or_compute_segments(self, pos):
        """Return (img, mask_skel, crossing_points, crossing_points_thick,
        segments) for a queue position, computing it once and caching --
        this is the expensive image-processing step, and both normal
        editing and "Edit" share this cache so it never runs twice
        for the same image."""
        if pos in self._segment_cache:
            return self._segment_cache[pos]
        fname = self.filenames[pos]
        real_basename = self._resolve_real_basename(fname)
        img = core.load_image(self.folder, real_basename)
        mask_skel, crossing_points, crossing_points_thick, segments = (
            core.prep_mask_and_segments(img)
        )
        result = (img, mask_skel, crossing_points, crossing_points_thick, segments)
        self._segment_cache[pos] = result
        return result

    def _load_at(self, pos):
        if pos < 0 or pos >= len(self.filenames):
            QMessageBox.information(self, "Done", "No more images in the queue.")
            return

        if 0 <= self.work_pos < len(self.statuses) and self.statuses[self.work_pos] == STATUS_CURRENT:
            self.statuses[self.work_pos] = STATUS_PENDING

        self.work_pos = pos
        fname = self.filenames[pos]
        self.current_fname = fname
        self._editing_from_review_pos = None

        result = self._safe_get_or_compute_segments(pos)
        if result is None:
            return
        (self.img, self.mask_skel, self.crossing_points,
         self.crossing_points_thick, self.segments) = result

        # Default is REMOVE: everything is excluded until explicitly
        # marked spiral / template / both.
        self.labels = np.full(len(self.segments), core.REMOVE, dtype=int)
        self.box_selection = []
        self.center_spiral = None
        self.center_template = None
        self.final_spiral = None
        self.final_template = None
        self.stage = "segment"
        self.awaiting_center = None

        self._set_controls_enabled(True)
        self.btn_pick_spiral_center.setEnabled(False)
        self.btn_pick_template_center.setEnabled(False)
        self.btn_save_next.setEnabled(False)
        self.btn_edit_review.setEnabled(False)
        self.btn_cancel_edit.setEnabled(False)

        self.statuses[pos] = STATUS_CURRENT
        self._rebuild_queue_list()
        self.queue_list.setCurrentRow(pos)

        done_count = sum(1 for s in self.statuses if s == STATUS_DONE)
        self.status_label.setText(
            f"{fname}  --  image {pos + 1} of {len(self.filenames)}  "
            f"({done_count} completed)"
        )

        self._redraw()

    def _load_review(self, pos):
        """Show a previously-completed image's saved result, read-only.
        Loaded straight from self.state -- no recomputation, so nothing
        here can accidentally overwrite the saved segmentation. Also
        warms the segment cache in the background so "Edit" (if
        used) doesn't have to re-run the image-processing pipeline."""
        if 0 <= self.work_pos < len(self.statuses) and self.statuses[self.work_pos] == STATUS_CURRENT:
            self.statuses[self.work_pos] = STATUS_PENDING

        self.work_pos = pos
        ii = pos + 1
        fname = self.filenames[pos]
        self.current_fname = fname
        self._editing_from_review_pos = None

        self.img = self.state["spiral_ims"][ii]
        self.final_spiral = self.state["spirals"][ii]
        self.final_template = self.state["template"][ii]
        self.center_spiral = self.state["center_spiral"][ii]
        self.center_template = self.state["center_template"][ii]
        self.stage = "review"

        # Force back to click mode and deactivate the drag-selectors --
        # they're separate matplotlib widgets, not gated by self.stage, so
        # leaving box/freehand active would let a drag during review
        # silently populate a selection against stale segment data.
        self.radio_click.setChecked(True)

        self._set_controls_enabled(False)
        self.btn_pick_spiral_center.setEnabled(False)
        self.btn_pick_template_center.setEnabled(False)
        self.btn_save_next.setEnabled(False)
        self.btn_edit_review.setEnabled(True)
        self.btn_cancel_edit.setEnabled(False)

        self._rebuild_queue_list()
        self.queue_list.setCurrentRow(pos)

        self.status_label.setText(
            f"{fname}  --  viewing saved result (read-only), image {pos + 1} "
            f"of {len(self.filenames)}. 'Edit' adjusts it starting from "
            f"labels matched to this saved result."
        )

        self._redraw()
        # Warm the segment cache for a possible Edit -- silent
        # best-effort only, since this runs just from viewing a review
        # (before the person has actually asked for Edit).
        # If resolution fails, review still works fine on its own (it
        # doesn't need segments); Edit will surface the same
        # error clearly and informatively if actually clicked.
        try:
            self._get_or_compute_segments(pos)
        except FileNotFoundError:
            pass

    def _load_preview(self, pos):
        """Raw-image-only look at a not-yet-reached pending image -- lets
        you see what's coming without being able to segment it out of
        turn. No processing is run (no skeleton/segments computed)."""
        if 0 <= self.work_pos < len(self.statuses) and self.statuses[self.work_pos] == STATUS_CURRENT:
            self.statuses[self.work_pos] = STATUS_PENDING

        self.work_pos = pos
        fname = self.filenames[pos]
        self.current_fname = fname
        self._editing_from_review_pos = None
        real_basename = self._resolve_real_basename(fname)

        try:
            self.img = core.load_image(self.folder, real_basename)
        except FileNotFoundError:
            QMessageBox.warning(self, "File not found",
                                 f"Could not find {real_basename}.png in {self.folder}.")
            return

        self.stage = "preview"
        self.radio_click.setChecked(True)  # reset mode, deactivate drag-selectors

        self._set_controls_enabled(False)
        self.btn_pick_spiral_center.setEnabled(False)
        self.btn_pick_template_center.setEnabled(False)
        self.btn_save_next.setEnabled(False)
        self.btn_edit_review.setEnabled(False)
        self.btn_cancel_edit.setEnabled(False)

        self._rebuild_queue_list()
        self.queue_list.setCurrentRow(pos)

        next_required = self._next_required_pos()
        next_name = self.filenames[next_required] if next_required is not None else None
        self.status_label.setText(
            f"{fname}  --  preview only (raw image, not yet segmented). "
            + (f"Complete images in order to reach it -- next up: {next_name}."
               if next_name else "")
        )

        self._redraw()

    def edit_from_review(self):
        """Enter editing from a completed image's review screen: guesses
        each segment's starting label by matching it against the
        currently-saved spiral/template result (best effort, not exact --
        see core.match_segments_to_saved), so you're adjusting from
        something close to what's already there instead of from scratch.
        Nothing is written to disk unless you go through the normal
        Accept -> pick centers -> Save flow; 'Cancel Edit' bails out with
        no changes at all."""
        if self.stage != "review":
            return
        pos = self.work_pos
        saved_final_spiral = self.final_spiral
        saved_final_template = self.final_template

        result = self._safe_get_or_compute_segments(pos)
        if result is None:
            return
        (self.img, self.mask_skel, self.crossing_points,
         self.crossing_points_thick, self.segments) = result

        self.labels = core.match_segments_to_saved(
            self.segments, saved_final_spiral, saved_final_template, self.img.shape
        )
        self.box_selection = []
        self.center_spiral = None
        self.center_template = None
        self.final_spiral = None
        self.final_template = None
        self.stage = "segment"
        self.awaiting_center = None
        self._editing_from_review_pos = pos

        self.radio_click.setChecked(True)
        self._set_controls_enabled(True)
        self.btn_pick_spiral_center.setEnabled(False)
        self.btn_pick_template_center.setEnabled(False)
        self.btn_save_next.setEnabled(False)
        self.btn_edit_review.setEnabled(False)
        self.btn_cancel_edit.setEnabled(True)

        self.statuses[pos] = STATUS_CURRENT
        self._rebuild_queue_list()
        self.queue_list.setCurrentRow(pos)

        self.status_label.setText(
            f"{self.filenames[pos]}  --  editing (labels matched from the "
            f"saved result, so it starts close to what you had). Adjust as "
            f"needed, then Accept + pick centers + Save to keep the "
            f"changes, or Cancel Edit to discard them and keep the "
            f"original saved result untouched."
        )

        self._preserve_view = False
        self._redraw()

    def cancel_edit_from_review(self):
        """Bail out of an Edit session with zero changes -- reloads the
        saved review state fresh, exactly as if Edit had never been
        clicked."""
        if self._editing_from_review_pos is None:
            return
        pos = self._editing_from_review_pos
        self._editing_from_review_pos = None
        self._load_review(pos)

    def next_image(self):
        # With strict in-order completion, the next required position is
        # always the right place to go next -- this also correctly
        # handles the "Edit" case (redoing an earlier image while
        # later ones are already done doesn't need to re-open them).
        pos = self._next_required_pos()
        if pos is None:
            self.stage = "segment"  # nothing left unsaved -- see save_and_next
            QMessageBox.information(self, "Done", "All images in this folder are complete.")
            self.status_label.setText("All images processed.")
            return
        self._load_at(pos)

    # ----------------------------------------------------------- Drawing --

    @staticmethod
    def _plot_target(ax, row, col, color):
        """Draw a target/reticle marker (ring + crosshair) at (row, col)
        rather than a plain '+' -- easier to spot and reads unambiguously
        as "this is the picked point", especially over busy line art."""
        ax.plot(col, row, marker="o", markersize=20, markerfacecolor="none",
                 markeredgecolor=color, markeredgewidth=2)
        ax.plot(col, row, marker="+", markersize=11,
                 color=color, markeredgewidth=2)

    def _redraw(self):
        # Preserve zoom/pan across a redraw if the caller asked for it
        # (an interaction on the SAME image -- label toggle, selection,
        # center pick). ax.clear() would otherwise silently reset the
        # view every single time, which is why zooming in used to feel
        # like it "snapped back out" on every click.
        preserve = self._preserve_view
        self._preserve_view = False
        if preserve:
            xlim_l, ylim_l = self.ax_left.get_xlim(), self.ax_left.get_ylim()
            xlim_r, ylim_r = self.ax_right.get_xlim(), self.ax_right.get_ylim()

        self.ax_left.clear()
        self.ax_right.clear()

        if self.stage == "segment":
            self.ax_left.imshow(self.img, cmap="gray")

            base = self.mask_skel & ~self.crossing_points_thick
            self.ax_right.imshow(base, cmap="gray")
            for i, coords in enumerate(self.segments):
                color = LABEL_COLORS.get(self.labels[i])
                if color:
                    self.ax_right.plot(coords[:, 1], coords[:, 0], ".",
                                        color=color, markersize=2)
            for i in self.box_selection:
                coords = self.segments[i]
                self.ax_right.plot(coords[:, 1], coords[:, 0], ".",
                                    color="cyan", markersize=3)
            self.ax_right.set_title(
                f"red=template  green=spiral  blue=both  "
                f"(unmarked=excluded)  cyan=selected "
                f"({len(self.segments)} segments)"
            )

        elif self.stage == "centers":
            self.ax_left.imshow(self.final_spiral_skel, cmap="gray")
            self.ax_left.set_title("SPIRAL -- pick center")
            if self.center_spiral is not None:
                self._plot_target(self.ax_left, *self.center_spiral, "red")

            self.ax_right.imshow(self.final_template_skel, cmap="gray")
            self.ax_right.set_title("TEMPLATE -- pick center")
            if self.center_template is not None:
                self._plot_target(self.ax_right, *self.center_template, "red")

        elif self.stage == "review":
            self.ax_left.imshow(self.img, cmap="gray")

            h, w = self.final_spiral.shape
            overlay = np.zeros((h, w, 3))
            overlay[..., 1] = self.final_spiral    # green channel = spiral
            overlay[..., 0] = self.final_template  # red channel = template
            self.ax_right.imshow(overlay)
            if self.center_spiral is not None:
                self._plot_target(self.ax_right, *self.center_spiral, "lime")
            if self.center_template is not None:
                self._plot_target(self.ax_right, *self.center_template, "red")
            self.ax_right.set_title(
                "Saved result (read-only) -- green=spiral  red=template  "
                "targets=centers"
            )

        elif self.stage == "preview":
            self.ax_left.imshow(self.img, cmap="gray")

            self.ax_right.text(
                0.5, 0.5, "Not yet segmented\n(complete earlier images first)",
                ha="center", va="center", transform=self.ax_right.transAxes,
                fontsize=11,
            )

        for ax in (self.ax_left, self.ax_right):
            ax.set_xticks([])
            ax.set_yticks([])

        if preserve:
            self.ax_left.set_xlim(xlim_l)
            self.ax_left.set_ylim(ylim_l)
            self.ax_right.set_xlim(xlim_r)
            self.ax_right.set_ylim(ylim_r)

        self.canvas.draw_idle()

    # ------------------------------------------------------------- Modes --

    def _on_mode_toggle(self):
        if self.radio_click.isChecked():
            self.mode = "click"
        elif self.radio_box.isChecked():
            self.mode = "box"
        else:
            self.mode = "freehand"
        self.rect_selector.set_active(self.mode == "box")
        self.lasso_selector.set_active(self.mode == "freehand")
        for b in (self.btn_mark_spiral, self.btn_mark_template,
                  self.btn_mark_remove, self.btn_mark_both):
            b.setEnabled(False)
        self.box_selection = []
        self._preserve_view = True
        self._redraw()

    def _on_box_select(self, eclick, erelease):
        if self.stage != "segment":
            return
        row_min, row_max = sorted([eclick.ydata, erelease.ydata])
        col_min, col_max = sorted([eclick.xdata, erelease.xdata])
        self.box_selection = core.segments_in_box(
            self.segments, row_min, row_max, col_min, col_max
        )
        enabled = len(self.box_selection) > 0
        for b in (self.btn_mark_spiral, self.btn_mark_template,
                  self.btn_mark_remove, self.btn_mark_both):
            b.setEnabled(enabled)
        self._preserve_view = True
        self._redraw()

    # Distance (in pixels) within which a segment point must fall of the
    # freehand-drawn path to count as "intersected" by it.
    LASSO_INTERSECT_THRESHOLD = 4.0

    def _on_lasso_select(self, verts):
        if self.stage != "segment":
            return
        verts = np.asarray(verts)
        if len(verts) < 2:
            return
        # verts are (x, y) i.e. (col, row); segments store (row, col)
        path_rc = np.column_stack([verts[:, 1], verts[:, 0]])
        self.box_selection = core.segments_near_path(
            self.segments, path_rc, threshold=self.LASSO_INTERSECT_THRESHOLD
        )
        enabled = len(self.box_selection) > 0
        for b in (self.btn_mark_spiral, self.btn_mark_template,
                  self.btn_mark_remove, self.btn_mark_both):
            b.setEnabled(enabled)
        self._preserve_view = True
        self._redraw()

    def _apply_bulk_label(self, label):
        for i in self.box_selection:
            self.labels[i] = label
        self.box_selection = []
        for b in (self.btn_mark_spiral, self.btn_mark_template,
                  self.btn_mark_remove, self.btn_mark_both):
            b.setEnabled(False)
        self._preserve_view = True
        self._redraw()

    def _deactivate_toolbar_tool(self):
        """Turn off the navigation toolbar's zoom/pan tool if it's active.
        Left on, it intercepts the next click as a zoom/pan drag instead
        of letting it register as a plain click -- which is why center
        picking (and segment clicking) could silently do nothing."""
        mode = getattr(self.toolbar, "mode", None)
        mode_str = getattr(mode, "name", str(mode)).upper() if mode else ""
        if "PAN" in mode_str:
            self.toolbar.pan()
        elif "ZOOM" in mode_str:
            self.toolbar.zoom()

    # Full cycle order used by click-to-toggle mode.
    CLICK_CYCLE = (core.REMOVE, core.SPIRAL, core.TEMPLATE, core.BOTH)

    # Radius (in pixels) removed around a manual split point -- needs to
    # be enough to actually break pixel-adjacency between the two strands
    # at a tangential contact, but small enough not to eat real curve.
    SPLIT_RADIUS = 2

    def _on_canvas_click(self, event):
        if event.inaxes is not self.ax_right and event.inaxes is not self.ax_left:
            return

        if self.stage == "centers" and self.awaiting_center is not None:
            if event.xdata is None or event.ydata is None:
                return  # click landed just outside the image bounds
            row, col = round(event.ydata), round(event.xdata)
            if self.awaiting_center == "spiral" and event.inaxes is self.ax_left:
                snapped = core.nearest_point_on_mask(self.final_spiral_skel, row, col)
                self.center_spiral = snapped if snapped is not None else (row, col)
                self.awaiting_center = None
                self._preserve_view = True
                self._redraw()
            elif self.awaiting_center == "template" and event.inaxes is self.ax_right:
                snapped = core.nearest_point_on_mask(self.final_template_skel, row, col)
                self.center_template = snapped if snapped is not None else (row, col)
                self.awaiting_center = None
                self._preserve_view = True
                self._redraw()
            self._update_save_button()
            return

        if self.stage != "segment" or event.inaxes is not self.ax_right:
            return
        if event.xdata is None or event.ydata is None:
            return

        row, col = event.ydata, event.xdata

        if self.mode != "click":
            return

        idx = core.nearest_segment_index(self.segments, row, col)
        if idx is None:
            return

        # Left click advances through the cycle, right click steps back --
        # lets you reach every state (spiral / template / remove / both)
        # and nudge past an overshoot without relooping the whole way.
        order = self.CLICK_CYCLE
        cur = self.labels[idx]
        pos = order.index(cur) if cur in order else 0
        step = -1 if event.button == 3 else 1
        self.labels[idx] = order[(pos + step) % len(order)]
        self._preserve_view = True
        self._redraw()

    def _split_segment_at(self, row, col):
        """Manually break one segment into independent pieces at a
        clicked point -- for crossings the automatic branch-point
        detector missed (most commonly a tangential "kiss" between two
        curves, which often has no distinguishing pixel pattern for a
        purely automatic detector to catch)."""
        idx = core.nearest_segment_index(self.segments, row, col)
        if idx is None:
            return
        coords = self.segments[idx]

        # Snap to the actual nearest pixel IN this segment, so the split
        # happens exactly on the curve rather than at a slightly-off click.
        dists = (coords[:, 0] - row) ** 2 + (coords[:, 1] - col) ** 2
        nearest_i = np.argmin(dists)
        split_row, split_col = coords[nearest_i]

        pieces = core.split_segment(coords, split_row, split_col, radius=self.SPLIT_RADIUS)
        if len(pieces) < 2:
            self.status_label.setText(
                "Nothing to split there -- click closer to the actual "
                "crossing/contact point."
            )
            return

        old_label = self.labels[idx]
        self.segments = self.segments[:idx] + pieces + self.segments[idx + 1:]
        self.labels = np.concatenate([
            self.labels[:idx],
            np.full(len(pieces), old_label, dtype=int),
            self.labels[idx + 1:],
        ])
        self.box_selection = []
        self._preserve_view = True
        self._redraw()
        self.status_label.setText(
            f"Split into {len(pieces)} independent pieces (both keep the "
            f"original label for now). Switch to Click mode to classify "
            f"them separately."
        )

    # --------------------------------------------------------- Fallback --

    def open_manual_dialog(self):
        if not self.segments:
            return
        dlg = ManualClassifyDialog(
            self, self.segments, self.labels, self.mask_skel, self.crossing_points_thick
        )
        dlg.exec_()
        self._preserve_view = True
        self._redraw()

    def open_split_dialog(self):
        if not self.segments:
            return
        # Snapshot so Cancel can fully discard whatever splits happened
        # during this dialog session -- splits apply live (so you can see
        # each one as you go), but nothing is permanent until Done.
        segments_backup = list(self.segments)
        labels_backup = self.labels.copy()

        dlg = SplitSegmentDialog(self)
        result = dlg.exec_()

        if result != QDialog.Accepted:
            self.segments = segments_backup
            self.labels = labels_backup
            self.box_selection = []

        self._preserve_view = True
        self._redraw()

    # ---------------------------------------------------------- Accept --

    def _is_last_remaining(self):
        """True if every OTHER position is already done -- i.e. finishing
        the current one completes the whole folder."""
        return all(
            s == STATUS_DONE for i, s in enumerate(self.statuses)
            if i != self.work_pos
        )

    def accept_segmentation(self):
        spiral, template = core.reconstruct_images(
            self.mask_skel.shape, self.segments, self.labels, self.crossing_points
        )
        # resize_binary_curve (rather than the plain resize_result) keeps
        # the thin skeleton curve from fragmenting into dashes during the
        # resize -- see its docstring in core.py.
        spiral_r = core.resize_binary_curve(spiral > 0.5, self.output_size)
        template_r = core.resize_binary_curve(template > 0.5, self.output_size)

        self.final_spiral = spiral_r
        self.final_template = template_r
        self.final_spiral_skel = spiral_r > 0.5   # already a clean 1px skeleton
        self.final_template_skel = template_r > 0.5

        self.stage = "centers"
        self.btn_pick_spiral_center.setEnabled(True)
        self.btn_pick_template_center.setEnabled(True)
        # "Save and Finish" when this is the last thing left to do -- it
        # writes straight to the final (non-tmp) .pkl either way, this is
        # just making that visible in the button label.
        self.btn_save_next.setText(
            "Save and Finish" if self._is_last_remaining() else "Save && Next"
        )
        self._redraw()

    def _start_center_pick(self, which):
        self.awaiting_center = which
        self._deactivate_toolbar_tool()
        self.status_label.setText(
            f"Click near the center of the {which} panel. "
            f"It will snap to closest point on {which}."
        )

    def _update_save_button(self):
        ready = self.center_spiral is not None and self.center_template is not None
        self.btn_save_next.setEnabled(ready)
        if ready:
            self.status_label.setText("Centers set. Ready to save.")

    # ------------------------------------------------------------- Save --

    def save_and_next(self):
        pos = self.work_pos
        ii = pos + 1  # 1-indexed, matches original convention
        fname = self.filenames[pos]

        self.state["spiral_ims"][ii] = self.img
        self.state["spirals"][ii] = self.final_spiral
        self.state["crossing_points"][ii] = self.crossing_points
        self.state["template"][ii] = self.final_template
        self.state["center_template"][ii] = self.center_template
        self.state["center_spiral"][ii] = self.center_spiral
        # Permanent record of exactly which real file this came from, so
        # there's no ambiguity later even if the index name gets renamed.
        real_basename = self.state.get("basename_by_index", {}).get(fname, fname)
        self.state.setdefault("source_filenames", {})[ii] = real_basename + ".png"

        tmp_path = os.path.join(self.save_dir, f"{self.save_tag}_spiral_tmp.pkl")
        with open(tmp_path, "wb") as f:
            pickle.dump(self.state, f)

        self.statuses[pos] = STATUS_DONE

        if all(s == STATUS_DONE for s in self.statuses):
            final_path = os.path.join(self.save_dir, f"{self.save_tag}_spiral.pkl")
            with open(final_path, "wb") as f:
                pickle.dump(self.state, f)
            os.remove(tmp_path)
            # Everything is safely on disk now -- reset stage away from
            # "centers" so the next navigation click doesn't think there's
            # still unsaved work to protect and wrongly ask to discard it.
            self.stage = "segment"
            self._rebuild_queue_list()
            QMessageBox.information(self, "Done", "All images in this folder are complete.")
            self.status_label.setText("All images processed.")
            return

        self.next_image()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
