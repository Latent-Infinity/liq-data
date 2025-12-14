"""Backwards-compatibility entry points should import cleanly."""

import importlib.util
from pathlib import Path


def test_cli_shim_module_loads() -> None:
    """Directly import the legacy cli.py shim to cover compatibility surface."""
    module_path = Path(__file__).parents[1] / "src" / "liq" / "data" / "cli.py"
    spec = importlib.util.spec_from_file_location("liq.data.cli_shim", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    assert hasattr(module, "app")
    assert hasattr(module, "main")


def test_cli_main_entrypoint_imports() -> None:
    """Importing the -m entrypoint module should not execute CLI."""
    import importlib

    mod = importlib.import_module("liq.data.cli.__main__")
    assert hasattr(mod, "main")
