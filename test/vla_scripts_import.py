import importlib.util
from pathlib import Path


def load_script_module(module_name: str, script_path: str):
    path = Path(script_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
