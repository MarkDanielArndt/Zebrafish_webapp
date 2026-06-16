"""
Check and install required packages into Slicer's bundled Python.

Usage (called once at extension startup):
    from ZebrafishAnalysisLib.dependency_installer import check_and_install
    check_and_install()  # no-op if everything already installed
"""

REQUIRED_PACKAGES = [
    "segmentation_models_pytorch",
    "timm",
    "scikit-image",
    "opencv-python-headless",
    "huggingface_hub",
    "openpyxl",
    "pytesseract",
]

TORCH_PACKAGES = ["torch", "torchvision"]
TORCH_INDEX    = "https://download.pytorch.org/whl/cpu"


def _is_importable(name: str) -> bool:
    import importlib.util
    import_name = {
        "scikit-image":                "skimage",
        "opencv-python-headless":      "cv2",
        "huggingface_hub":             "huggingface_hub",
        "segmentation_models_pytorch": "segmentation_models_pytorch",
    }.get(name, name)
    return importlib.util.find_spec(import_name) is not None


def _numpy_major() -> int:
    """Return installed numpy major version, or 0 on failure."""
    try:
        import numpy as np
        return int(np.__version__.split(".")[0])
    except Exception:
        return 0


def check_and_install(show_restart_message: bool = True) -> None:
    """Install missing dependencies via slicer.util.pip_install."""
    try:
        import slicer
        import qt
    except ImportError:
        return  # running outside Slicer (e.g. unit tests) — skip

    missing_torch   = [p for p in TORCH_PACKAGES    if not _is_importable(p)]
    missing_general = [p for p in REQUIRED_PACKAGES if not _is_importable(p)]

    # numpy check deferred until after all installs — torch can pull numpy back up
    needs_any_install = bool(missing_torch or missing_general)
    if not needs_any_install and _numpy_major() < 2:
        slicer.util.showStatusMessage("Dependencies OK.")
        return

    pkg_list = (["torch", "torchvision"] if missing_torch else []) + missing_general + ["numpy<2"]
    pkg_lines = "\n".join(f"  • {p}" for p in pkg_list)
    confirm = qt.QMessageBox()
    confirm.setWindowTitle("ZebrafishAnalysis — Dependencies Required")
    confirm.setText(
        "The following packages must be installed into Slicer's Python environment "
        "before ZebrafishAnalysis can run:\n\n"
        + pkg_lines +
        "\n\nInstallation may take several minutes and requires an internet connection. "
        "Slicer must be restarted afterwards.\n\nInstall now?"
    )
    confirm.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
    confirm.setDefaultButton(qt.QMessageBox.Yes)
    if confirm.exec_() != qt.QMessageBox.Yes:
        slicer.util.showStatusMessage("ZebrafishAnalysis: dependency installation cancelled.")
        return

    steps = (["torch+torchvision"] if missing_torch else []) + \
            missing_general + \
            ["numpy<2"]  # always pin last so torch can't overwrite it
    total = len(steps)

    progress = qt.QProgressDialog(
        "Installing dependencies…", None, 0, total
    )
    progress.setWindowTitle("ZebrafishAnalysis — First Run Setup")
    progress.setMinimumWidth(400)
    progress.setWindowModality(qt.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()
    slicer.app.processEvents()

    step = 0

    if missing_torch:
        progress.setLabelText(f"Installing PyTorch (CPU) — may take several minutes… ({step + 1}/{total})")
        progress.setValue(step)
        slicer.app.processEvents()
        slicer.util.pip_install("torch torchvision --index-url " + TORCH_INDEX)
        step += 1

    for pkg in missing_general:
        progress.setLabelText(f"Installing {pkg}… ({step + 1}/{total})")
        progress.setValue(step)
        slicer.app.processEvents()
        slicer.util.pip_install(pkg)
        step += 1

    # pin numpy<2 after all other installs to prevent torch from pulling it back up
    progress.setLabelText(f"Pinning NumPy<2 for PyTorch compatibility… ({step + 1}/{total})")
    progress.setValue(step)
    slicer.app.processEvents()
    slicer.util.pip_install('"numpy<2"')
    step += 1

    progress.setValue(total)
    progress.close()

    slicer.util.showStatusMessage("Dependencies installed — restart required.")
    if show_restart_message:
        slicer.util.messageBox(
            "Required packages have been installed.\n"
            "Please restart 3D Slicer to complete setup."
        )
