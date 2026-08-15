from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirements():
    return [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_comfy_manager_has_no_redundant_install_script():
    assert not (REPO_ROOT / "install.py").exists()


def test_declared_dependencies_stay_aligned_and_exclude_optional_cupy():
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        project_dependencies = tomllib.load(file)["project"]["dependencies"]

    requirements = _requirements()
    assert requirements == project_dependencies
    assert not any("cupy" in dependency.lower() for dependency in requirements)
