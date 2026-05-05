"""Robust Sleep-EDF cassette downloader with retries and resume support."""

import argparse
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import requests
import shutil
import subprocess
import shlex

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

def _load_sleep_records(subject_count: int, subject_start: int = 0) -> tuple[str, List[Dict[str, str]]]:
    """Load canonical Sleep-EDF records from MNE's metadata CSV."""
    from mne.datasets.sleep_physionet import age

    # Try HTTP mirror first, if available; fallback to HTTPS
    base_url = "http://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
    records_path = Path(age.AGE_SLEEP_RECORDS)
    if not records_path.exists():
        raise FileNotFoundError(f"Could not find records file: {records_path}")

    selected_subjects = set(range(subject_start, subject_start + subject_count))
    rows: List[Dict[str, str]] = []
    with open(records_path, "r", encoding="utf-8") as f:
        header = next(f)
        _ = header
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            subject = int(parts[0])
            if subject not in selected_subjects:
                continue
            rows.append(
                {
                    "subject": str(subject),
                    "record": parts[1],
                    "kind": parts[2],
                    "sha1": parts[6],
                    "fname": parts[7],
                }
            )
    return base_url, rows


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_with_retries(url: str, dest: Path, expected_sha1: str, retries: int, timeout_sec: int) -> bool:
    """Stream download with retry/backoff; verify SHA1 if available."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and expected_sha1 and _sha1(dest) == expected_sha1:
        log.info("  skip (verified): %s", dest.name)
        return True

    for attempt in range(1, retries + 1):
        try:
            if dest.exists():
                # Once we have a final file on disk, trust it and skip further work.
                if expected_sha1 and _sha1(dest) == expected_sha1:
                    log.info("  skip (verified): %s", dest.name)
                    return True
                dest.unlink(missing_ok=True)

            headers = {}
            if tmp.exists():
                headers["Range"] = f"bytes={tmp.stat().st_size}-"

            # Use shorter connect timeout (10s) to fail fast on unreachable hosts
            connect_timeout = min(10, max(3, timeout_sec // 6))
            with requests.get(url, stream=True, timeout=(connect_timeout, timeout_sec), headers=headers) as resp:
                if resp.status_code == 416:
                    # Range not satisfiable, usually because the partial file is stale.
                    tmp.unlink(missing_ok=True)
                    time.sleep(min(30, 2 * attempt))
                    continue

                if resp.status_code not in (200, 206):
                    log.warning("  HTTP %s for %s", resp.status_code, url)
                    time.sleep(min(30, 2 * attempt))
                    continue

                mode = "ab" if resp.status_code == 206 and tmp.exists() else "wb"
                with open(tmp, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            f.write(chunk)

            if dest.exists():
                dest.unlink()
            tmp.replace(dest)
            if expected_sha1:
                got = _sha1(dest)
                if got != expected_sha1:
                    log.warning("  checksum mismatch for %s (got=%s exp=%s)", dest.name, got[:8], expected_sha1[:8])
                    dest.unlink(missing_ok=True)
                    time.sleep(min(30, 2 * attempt))
                    continue
            return True
        except Exception as exc:
            log.warning("  retry %d/%d for %s: %s", attempt, retries, dest.name, exc)
            time.sleep(min(30, 2 * attempt))
    return False


def _download_with_curl(url: str, dest: Path, expected_sha1: str, retries: int, timeout_sec: int) -> bool:
    """Fallback to system curl which may behave better through proxies and resume."""
    curl = shutil.which("curl")
    if not curl:
        log.info("  curl not available")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Use a single .part file for resume
    tmp = dest.with_suffix(dest.suffix + ".part")
    args = [
        curl,
        "-L",
        "--retry", str(retries),
        "--retry-all-errors",
        "--connect-timeout", str(max(30, int(timeout_sec / 10))),
        "--max-time", str(timeout_sec),
        "-C", "-",
        "-o", str(tmp),
        url,
    ]

    env = None
    # Let curl inherit HTTP(S)_PROXY from environment; user can set proxies externally.
    try:
        log.info("  curl -> %s", dest.name)
        proc = subprocess.run(args, check=False, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            log.warning("  curl failed (%d): %s", proc.returncode, proc.stderr.strip()[:200])
            return False
        # Move tmp to final dest
        if dest.exists():
            dest.unlink(missing_ok=True)
        tmp.replace(dest)
        if expected_sha1:
            got = _sha1(dest)
            if got != expected_sha1:
                log.warning("  curl checksum mismatch for %s (got=%s exp=%s)", dest.name, got[:8], expected_sha1[:8])
                dest.unlink(missing_ok=True)
                return False
        return True
    except Exception as exc:
        log.warning("  curl exception for %s: %s", dest.name, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/sleep_edf", help="Output directory")
    parser.add_argument("--subjects", type=int, default=10, help="Number of subjects (max 20)")
    parser.add_argument("--start-subject", type=int, default=0, help="First subject index to include")
    parser.add_argument("--retries", type=int, default=6, help="Retries per file")
    parser.add_argument("--timeout", type=int, default=600, help="Read timeout seconds")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = min(args.subjects, 20 - args.start_subject)
    base_url, rows = _load_sleep_records(n_subjects, subject_start=max(0, args.start_subject))
    log.info(
        "Downloading %d files for %d subjects starting at %d -> %s",
        len(rows),
        n_subjects,
        args.start_subject,
        out_dir,
    )

    ok, fail = 0, 0

    def _run(row: Dict[str, str]) -> bool:
        fname = row["fname"]
        url = f"{base_url}/{fname}"
        dest = out_dir / fname
        log.info("  down %s", dest.name)
        # Try requests first, fall back to curl if it fails
        if _download_with_retries(url, dest, row["sha1"], args.retries, args.timeout):
            return True
        log.info("  requests failed, trying curl for %s", dest.name)
        return _download_with_curl(url, dest, row["sha1"], args.retries, args.timeout)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(_run, row): row for row in rows}
        for future in as_completed(future_map):
            row = future_map[future]
            try:
                if future.result():
                    ok += 1
                else:
                    fail += 1
            except Exception as exc:
                fail += 1
                log.warning("  failed %s: %s", row["fname"], exc)

    log.info("Done. %d downloaded, %d failed.", ok, fail)
    if fail > 0:
        log.warning("Some files failed. Check subject ID range or network access.")
        log.warning("Manual download: https://physionet.org/content/sleep-edfx/1.0.0/")
    else:
        log.info("Next step:")
        log.info("  python -m sleep_staging.cli build-manifest \\")
        log.info("      --raw-dir %s \\", out_dir)
        log.info("      --manifest data/manifests/sleep_edf.csv")


if __name__ == "__main__":
    main()
