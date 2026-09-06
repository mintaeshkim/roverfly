from roverfly.paths import asset_path, resolve_model_path


def test_packaged_asset_exists() -> None:
    path = asset_path("quadrotor_falcon.xml")
    assert path.is_file()
    assert resolve_model_path(None, default=path.name) == str(path)


def test_legacy_asset_spelling_resolves_to_package() -> None:
    resolved = resolve_model_path("../assets/quadrotor_mini.xml", default="unused.xml")
    assert resolved.endswith("roverfly/assets/quadrotor_mini.xml")
