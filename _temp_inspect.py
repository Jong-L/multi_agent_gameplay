"""Check ModelCatalog.get_model_v2 source and RLlib config."""
import sys, pathlib, inspect
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "Python" / "training"))

from ray.rllib.models.catalog import ModelCatalog

src = inspect.getsource(ModelCatalog.get_model_v2)
# Print the section near where model_cls is instantiated
for i, line in enumerate(src.split("\n")):
    if any(kw in line for kw in ["model_cls", "instance =", "custom_model", "**", "kwargs"]):
        print(f"  L{i:4d}: {line}")
