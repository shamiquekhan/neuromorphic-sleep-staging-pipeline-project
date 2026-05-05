"""Download Sleep-EDF cassette data using MNE's built-in fetcher.

This is a fallback for the remaining subjects when the direct downloader
gets stuck on partial files or transient PhysioNet errors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mne


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/sleep_edf")
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--start-subject", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_subject = min(args.start_subject + args.subjects, 20)
    subjects = list(range(max(0, args.start_subject), end_subject))
    print(f"Fetching subjects: {subjects}")
    files = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=subjects,
        recording=[1, 2],
        path=str(out_dir),
        on_missing="warn",
    )
    flat = [str(p) for pair in files for p in pair]
    print(f"Downloaded {len(flat)} files")
    for path in flat:
        print(path)


if __name__ == "__main__":
    main()
