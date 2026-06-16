"""
Results tab — QTableWidget showing all measurements.
"""

import qt


COLUMNS = [
    ("Filename",              "filename",     str),
    ("Length (µm)",           "length",       lambda v: f"{v:.1f}" if v is not None else ""),
    ("Curvature class",       "curvature",    lambda v: str(v) if v is not None else ""),
    ("Length/straight ratio", "ratio",        lambda v: f"{v:.3f}" if v is not None else ""),
    ("Eye area (µm²)",        "eye_area",     lambda v: f"{v:.1f}" if v is not None else ""),
    ("Eye diameter (µm)",     "eye_diameter", lambda v: f"{v:.1f}" if v is not None else ""),
    ("Error",                 "error",        lambda v: v or ""),
]


class ResultsTab(qt.QWidget):
    def __init__(self):
        super().__init__()
        self._table = qt.QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        self._table.horizontalHeader().setSectionResizeMode(
            0, qt.QHeaderView.Stretch
        )
        self._table.editTriggers = qt.QAbstractItemView.NoEditTriggers
        self._table.selectionBehavior = qt.QAbstractItemView.SelectRows

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self._table)

    def populate(self, results) -> None:
        n = len(results)
        self._table.rowCount = n
        for row in range(n):
            r = results[row]
            for col in range(len(COLUMNS)):
                _, key, fmt = COLUMNS[col]
                val = r.get(key)
                self._table.setItem(row, col, qt.QTableWidgetItem(fmt(val)))
