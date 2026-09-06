"""Filesystem paths owned by the package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = PACKAGE_DIR / "assets"
MODEL_DIR = PACKAGE_DIR / "models"


def asset_path(name: str) -> Path:
    """Return an existing packaged MuJoCo asset path."""
    path = ASSET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Unknown RoVerFly asset: {name}")
    return path


def resolve_model_path(path: str | Path | None, *, default: str) -> str:
    """Resolve a user model path, falling back to a packaged asset."""
    if path is None:
        return str(asset_path(default))

    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    # Accept the old ``../assets/model.xml`` spelling after package migration.
    packaged = ASSET_DIR / candidate.name
    if packaged.is_file():
        return str(packaged)
    raise FileNotFoundError(f"MuJoCo model does not exist: {candidate}")
