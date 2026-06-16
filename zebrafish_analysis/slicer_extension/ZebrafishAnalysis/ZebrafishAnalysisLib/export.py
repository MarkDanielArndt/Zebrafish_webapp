"""
Export analysis results to Excel or CSV.

Adding a new format: add a function export_<fmt>(results, path) here
and wire it to a new button in widget.py.
"""

HEADERS = [
    ("Filename",              "filename"),
    ("Length (µm)",           "length"),
    ("Curvature class",       "curvature"),
    ("Length/straight ratio", "ratio"),
    ("Eye area (µm²)",        "eye_area"),
    ("Eye diameter (µm)",     "eye_diameter"),
    ("Error",                 "error"),
]


def export_excel(results: list, path: str) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Zebrafish Results"
    ws.append([h for h, _ in HEADERS])
    for r in results:
        ws.append([r.get(k) for _, k in HEADERS])
    wb.save(path)


def export_csv(results: list, path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([h for h, _ in HEADERS])
        for r in results:
            w.writerow([r.get(k) for _, k in HEADERS])
