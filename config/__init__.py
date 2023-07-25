# __init__.py

import pathlib
import tomli

path = pathlib.Path(__file__).parent / "literature.toml"
with path.open(mode="rb") as fp:
    literature = tomli.load(fp)

path = pathlib.Path(__file__).parent / "device.toml"
with path.open(mode="rb") as fp:
    device = tomli.load(fp)
