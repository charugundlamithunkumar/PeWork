from pathlib import Path


def default_package():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "Person A"
        if (candidate / "models/person_a/model.json").is_file():
            return candidate
    return Path("../Person A")
