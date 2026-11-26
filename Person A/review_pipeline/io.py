"""Run configuration and artifact I/O shared by the seven notebooks."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_DATASET_SUBDIR = Path("datasets") / "ECommerce_Project"
RAW_INPUT_NAMES = ("amazon_data.csv", "metadata", "reviewContent")


@dataclass(frozen=True)
class Run:
    project: Path
    run_id: str
    fixture: bool = False
    device: str = "cpu"
    seed: int = 42
    dataset_dir: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "project", Path(self.project))
        if self.dataset_dir is not None:
            object.__setattr__(self, "dataset_dir", Path(self.dataset_dir).expanduser().resolve())
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_id):
            raise ValueError("Use letters, digits, underscores or hyphens in RUN_ID.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("DEVICE must be cpu or cuda.")

    @property
    def root(self):
        return self.project / "v2" / ("fixtures" if self.fixture else "runs") / self.run_id

    @property
    def raw(self):
        if self.fixture:
            return self.root / "fixture_inputs"
        return self.dataset_dir or self.project / DEFAULT_DATASET_SUBDIR

    @property
    def folds(self):
        return 3 if self.fixture else 5

    def start(self, stage):
        self.root.mkdir(parents=True, exist_ok=True)
        if stage <= 5 and (self.root / "tests_opened.json").exists():
            raise ValueError("This run has opened its test sets. Use a new RUN_ID and disclose any subsequent changes.")
        config = self.root / "run_config.json"
        current = {**asdict(self), "project": str(self.project.resolve())}
        current["dataset_dir"] = str(self.dataset_dir) if self.dataset_dir else None
        if config.exists():
            previous = self.read_json("run_config.json")
            previous.setdefault("dataset_dir", None)
            if previous != current:
                raise ValueError("Run settings changed. Choose a new RUN_ID.")
        if stage > 1:
            self.require(stage - 1)
        self.write_json("run_config.json", current)
        for n in range(stage, 8):
            (self.root / f"stage_{n:02d}.json").unlink(missing_ok=True)

    def finish(self, stage):
        self.write_json(f"stage_{stage:02d}.json", {"stage": stage, "fixture": self.fixture})

    def require(self, stage):
        if not (self.root / f"stage_{stage:02d}.json").exists():
            raise FileNotFoundError(f"Run notebook {stage:02d} first for {self.root}.")

    def write_json(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def read_json(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def write_table(self, name, frame):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)

    def read_table(self, name):
        path = self.root / name
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    checksum = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def provenance():
    versions = {}
    for name in ("pandas", "numpy", "scikit-learn", "xgboost", "pyarrow", "vaderSentiment"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "dependencies": versions,
            "source_hashes": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


def fixture_inputs(root, seed):
    """Create separate, labelled toy source files with overlapping product histories."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    amazon = []
    metadata, content = [], []
    start = pd.Timestamp("2020-01-01")
    for user in range(150):
        for offset in range(12):
            product = (user * 7 + offset * 3) % 37
            date = (start + pd.Timedelta(days=int(user + offset * 5))).strftime("%Y-%m-%d")
            suspicious = rng.random() < 0.32
            rating = int(rng.choice([4, 5] if suspicious else [1, 2, 3, 4, 5]))
            message = "Best purchase ever! Highly recommend!!" if suspicious else "The package arrived and fit well."
            review = f"{message} Item {product}."
            amazon.append({
                "review_id": f"a{user}-{offset}", "reviewer_id": f"a{user}", "asin": f"p{product}",
                "review_date": date, "review_rating": rating, "review_text": review,
                "fake_review_product": 1, "reviewer_classified_fake": int(suspicious),
                "review_is_removed_by_amazon": int(rng.random() < (0.45 if suspicious else 0.03)),
            })
            if user < 65:
                yelp_rating = rating if offset % 3 else int(rng.integers(1, 4))
                metadata.append([f"y{user}", f"yp{product}", yelp_rating, -1 if suspicious else 1, date])
                content.append([f"y{user}", f"yp{product}", date, review])
    pd.DataFrame(amazon).to_csv(root / "amazon_data.csv", index=False)
    pd.DataFrame(metadata).to_csv(root / "metadata", sep="\t", header=False, index=False)
    pd.DataFrame(content).to_csv(root / "reviewContent", sep="\t", header=False, index=False)
