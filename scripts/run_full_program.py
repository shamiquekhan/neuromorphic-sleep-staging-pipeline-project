from __future__ import annotations

import argparse
from pathlib import Path

from sleep_staging.pipeline import FullPipelineConfig, run_full_program


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Sleep-EDF pipeline end-to-end")
    parser.add_argument("--raw-dir", default=str(FullPipelineConfig().raw_dir), help="Folder containing PSG/Hypnogram EDF files")
    parser.add_argument("--manifest", default=str(FullPipelineConfig().manifest))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", default=str(FullPipelineConfig().cache_dir))
    parser.add_argument("--artifacts-dir", default=str(FullPipelineConfig().artifacts_dir))
    parser.add_argument("--firmware-dir", default=str(FullPipelineConfig().firmware_dir))
    args = parser.parse_args()

    config = FullPipelineConfig(
        raw_dir=Path(args.raw_dir),
        manifest=Path(args.manifest),
        epochs=args.epochs,
        batch_size=args.batch_size,
        artifacts_dir=Path(args.artifacts_dir),
        firmware_dir=Path(args.firmware_dir),
        cache_dir=Path(args.cache_dir),
    )
    run_full_program(config)


if __name__ == "__main__":
    main()
