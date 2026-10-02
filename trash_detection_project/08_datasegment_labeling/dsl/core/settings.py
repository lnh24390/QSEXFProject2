"""Application settings, persisted to configs/settings.json."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


def _app_dir() -> Path:
    """The directory the app owns: weights/, configs/, datasets/, runs/.

    PyInstaller rewrites `__file__` to point inside the bundle, so
    `parents[2]` would land in a temp extraction directory that is wiped on
    exit. A frozen build must keep its data beside the executable instead --
    that is also where the installer creates weights/.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


APP_DIR = _app_dir()
CONFIG_DIR = APP_DIR / "configs"
SETTINGS_FILE = CONFIG_DIR / "settings.json"


def train_source_root() -> Path:
    """Directory to put on PYTHONPATH so a subprocess can `import dsl`.

    Training always runs in a separate process (DEVELOPMENT.md 3.6). From source
    that process is the same interpreter and the repo root is enough. A
    frozen build cannot do this at all on its own -- the exe rejects `-m`,
    and torch is not bundled -- so it hands the job to an external
    interpreter, which needs the plain .py sources that build.spec ships
    alongside the executable.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", "") or Path(sys.executable).parent)
        return base / "dsl_src"
    return APP_DIR


def resolve_under_app(value: str | Path) -> Path:
    """Absolute paths win; relative ones hang off APP_DIR, never the CWD.

    Settings store relative names ("weights") so a project stays portable,
    but resolving them against the process CWD would put the folder wherever
    the app happened to be launched from.
    """
    p = Path(value)
    return p if p.is_absolute() else APP_DIR / p


@dataclass
class Settings:
    # ---- memory budget (see DEVELOPMENT.md section 3) --------------------------
    image_cache_mb: int = 1024      # full-resolution image LRU budget
    thumb_cache_items: int = 512    # decoded thumbnails kept in RAM
    thumb_size: int = 96            # px, long edge
    file_view_mode: str = "썸네일"    # 썸네일 | 목록 | 폴더
    prefetch_neighbors: int = 1     # images preloaded ahead/behind
    sam_max_side: int = 0           # 0 = feed SAM the full resolution image
    log_max_lines: int = 5000

    # ---- labelling behaviour ---------------------------------------------
    auto_save: bool = True
    auto_advance: bool = False      # jump to next unlabeled after save
    hide_completed: bool = False    # drop finished images from the list
    poly_epsilon: float = 0.002     # contour simplification ratio
    poly_smooth: int = 2            # contour moving-average radius, 0 = raw staircase
    min_area: float = 24.0          # ignore specks smaller than this
    keep_largest_only: bool = False
    fill_alpha: int = 70            # 0-255 mask fill opacity, 0 = outline only
    outline_width: float = 1.8      # shape stroke width in screen px

    # ---- models -----------------------------------------------------------
    device: str = "auto"            # auto | cuda | cpu | mps
    sam_backend: str = "sam2.1_hiera_small"
    sam_precision: str = "fp16"     # clamped to the spec's quant_modes on load
    weights_dir: str = "weights"
    # Interpreter that runs training. Empty = this one, which is right from
    # source but impossible in a frozen build (the exe rejects `-m`, and torch
    # is not bundled), so a packaged app needs an external python here.
    train_python: str = ""

    # ---- misc -------------------------------------------------------------
    recent_projects: List[str] = field(default_factory=list)
    last_export_dir: str = "datasets"
    export_max_side: int = 0        # 0 = export images at original size
    theme: str = "dark"

    def __post_init__(self) -> None:
        # Where this instance came from / goes back to. Deliberately not a
        # dataclass field, so it never lands in the JSON. Tests and embedders
        # can point an instance at their own file instead of the global one.
        self.config_path: Path = SETTINGS_FILE

    # ---- helpers ----------------------------------------------------------
    @property
    def image_cache_bytes(self) -> int:
        return max(64, int(self.image_cache_mb)) * 1024 * 1024

    @property
    def weights_path(self) -> Path:
        """weights/ as an absolute path.

        Deliberately does not create anything: this is read during
        MainWindow.__init__, and a stale config pointing at a vanished drive
        must not take the app down with an OSError. Writers call
        `ensure_weights_dir()` when they actually need the directory.
        """
        return resolve_under_app(self.weights_dir)

    def ensure_weights_dir(self) -> Path:
        """weights/, created. Falls back to the default if the config is stale."""
        p = self.weights_path
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            fallback = APP_DIR / "weights"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    @property
    def export_path(self) -> Path:
        return resolve_under_app(self.last_export_dir)

    @property
    def runs_path(self) -> Path:
        return resolve_under_app("runs")

    def push_recent(self, path: str, limit: int = 10) -> None:
        path = str(path)
        self.recent_projects = [path] + [p for p in self.recent_projects
                                         if p != path]
        del self.recent_projects[limit:]

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic, like every other writer in the project
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)
        self.config_path = path
        return path

    @staticmethod
    def load(path: Path | None = None) -> "Settings":
        path = Path(path or SETTINGS_FILE)
        s = Settings()
        s.config_path = path
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return s
            valid = {f for f in Settings.__dataclass_fields__}
            for k, v in d.items():
                if k in valid:
                    setattr(s, k, v)
        return s


def resolve_device(pref: str = "auto") -> str:
    """Pick a torch device string without importing torch at module level."""
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
