"""
Exclude tab — checkboxes to mark images for exclusion from export.

on_change callback(excluded_set) is called whenever the set changes.
"""

import qt


class ExcludeTab(qt.QWidget):
    def __init__(self, on_change):
        super().__init__()
        self._on_change = on_change
        self._checkboxes = []

        self._table = qt.QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Filename", "Exclude"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, qt.QHeaderView.Stretch
        )
        self._table.editTriggers = qt.QAbstractItemView.NoEditTriggers

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self._table)

    def populate(self, results) -> None:
        self._table.rowCount = len(results)
        self._checkboxes = []

        for row in range(len(results)):
            r = results[row]
            self._table.setItem(row, 0, qt.QTableWidgetItem(r["filename"]))

            chk = qt.QCheckBox()
            chk.setChecked(bool(r.get("error")))

            filename = r["filename"]
            chk.toggled.connect(lambda _checked, _fn=filename: self._notify())
            self._table.setCellWidget(row, 1, chk)
            self._checkboxes.append((filename, chk))

        self._notify()  # propagate initial checked state (error rows pre-checked)

    def _notify(self):
        excluded = {fn for fn, chk in self._checkboxes if chk.isChecked()}
        self._on_change(excluded)

    def get_excluded(self) -> set:
        return {fn for fn, chk in self._checkboxes if chk.isChecked()}
