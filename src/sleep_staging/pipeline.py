from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .cli import cmd_build_all, cmd_build_manifest


DEFAULT_KAGGLE_SLEEP_EDF = Path(
    r"C:\Users\shami\.cache\kagglehub\datasets\iamommpatel\physiobank-database-sleep-edfx-cassette\versions\1\physiobank_database_sleep-edfx_sleep-cassette"
)
DEFAULT_FULL_MANIFEST = Path("data/manifests/sleep_edf_full.csv")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_FIRMWARE_DIR = Path("firmware/src")
DEFAULT_CACHE_DIR = Path("data/cache")


@dataclass(frozen=True)
class FullPipelineConfig:
    raw_dir: Path = DEFAULT_KAGGLE_SLEEP_EDF
    manifest: Path = DEFAULT_FULL_MANIFEST
    epochs: int = 30
    batch_size: int = 16
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    firmware_dir: Path = DEFAULT_FIRMWARE_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR

    @property
    def teacher_ckpt(self) -> Path:
        return self.artifacts_dir / "teacher.pt"

    @property
    def student_ckpt(self) -> Path:
        return self.artifacts_dir / "student.pt"


def build_full_manifest(config: FullPipelineConfig) -> None:
    args = SimpleNamespace(raw_dir=str(config.raw_dir), manifest=str(config.manifest))
    cmd_build_manifest(args)


def run_full_real_pipeline(config: FullPipelineConfig) -> None:
    args = SimpleNamespace(
        mode="real",
        manifest=str(config.manifest),
        epochs=config.epochs,
        batch_size=config.batch_size,
        teacher_ckpt=str(config.teacher_ckpt),
        student_ckpt=str(config.student_ckpt),
        artifacts_dir=str(config.artifacts_dir),
        firmware_dir=str(config.firmware_dir),
        cache_dir=str(config.cache_dir),
    )
    cmd_build_all(args)


def run_full_program(config: FullPipelineConfig = FullPipelineConfig()) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.manifest.parent.mkdir(parents=True, exist_ok=True)
    build_full_manifest(config)
    run_full_real_pipeline(config)
