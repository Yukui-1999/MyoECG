"""Prepare durable training manifests with metadata or full NPZ validation."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Callable, Optional
import zipfile

from .dataset import PathLike, ShortAxisNPZDataset


def ensure_nonempty_manifest(
    data_root: PathLike,
    manifest: PathLike,
    progress: Optional[Callable] = None,
) -> Path:
    """Reuse a cohort, or atomically create one excluding zero-byte NPZ files.

    Existing manifests are never rescanned or rewritten. For generated manifests,
    the adjacent audit records the resolved data root and prevents accidental
    reuse with a different root. Discovery follows ``ShortAxisNPZDataset``;
    preparation checks only regular-file type and byte size, never NPZ contents.
    Other filesystem errors abort preparation instead of shrinking the cohort.
    """
    data_root = Path(data_root).expanduser()
    manifest = Path(manifest).expanduser()
    if progress is not None:
        progress("manifest_prepare", current_path=str(manifest), data_root=str(data_root))
    data_root = data_root.resolve()
    manifest = manifest.resolve()
    audit = manifest.with_suffix(".audit.json")
    try:
        manifest_stat = manifest.stat()
    except FileNotFoundError:
        manifest_stat = None
    if manifest_stat is not None:
        if not stat.S_ISREG(manifest_stat.st_mode):
            raise ValueError("manifest is not a regular file: {}".format(manifest))
        try:
            with audit.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except FileNotFoundError:
            pass
        else:
            recorded_root = report.get("data_root") if isinstance(report, dict) else None
            if recorded_root != str(data_root):
                raise ValueError(
                    "manifest audit data_root mismatch: {} records {!r}, requested {}. "
                    "Use a separate manifest for this data root.".format(
                        audit, recorded_root, data_root))
        if progress is not None:
            progress("manifest_ready", current_path=str(manifest), reused=True)
        return manifest

    dataset = ShortAxisNPZDataset(data_root, progress=progress)
    files = sorted(dataset.canonical_files)
    kept = []
    empty = []
    for index, path in enumerate(files):
        if progress is not None:
            progress("manifest_audit", current_path=str(path), total=len(files),
                     checked=index, kept=len(kept), empty=len(empty))
        try:
            metadata = path.stat()
        except OSError as error:
            raise OSError("Cannot stat NPZ sample {}: {}".format(path, error)) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("NPZ sample is not a regular file: {}".format(path))
        if metadata.st_size == 0:
            empty.append(str(path))
        else:
            kept.append(str(path))
    if not kept:
        raise ValueError("no nonempty NPZ samples were found in {} ({} empty files)".format(
            data_root, len(empty)))

    report = {
        "data_root": str(data_root),
        "total": len(files),
        "kept": len(kept),
        "empty": empty,
        "validation_scope": "size/type only; no full data validation",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(manifest.parent),
            prefix=".{}-".format(manifest.name), suffix=".tmp", delete=False,
        ) as handle:
            temporary_manifest = Path(handle.name)
            temporary_paths.append(temporary_manifest)
            handle.write("\n".join(kept) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(audit.parent),
            prefix=".{}-".format(audit.name), suffix=".tmp", delete=False,
        ) as handle:
            temporary_audit = Path(handle.name)
            temporary_paths.append(temporary_audit)
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # The manifest is the completion marker: consumers must never observe it
        # before its complete audit report has been published.
        os.replace(str(temporary_audit), str(audit))
        os.replace(str(temporary_manifest), str(manifest))
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if progress is not None:
        progress("manifest_ready", current_path=str(manifest), audit_path=str(audit),
                 reused=False, total=len(files), kept=len(kept), empty=len(empty))
    return manifest


VALIDATION_VERSION = "sa_npz_arrays_v1"
VALIDATION_SCOPE = (
    "required NPZ image/slice_valid_mask full decode and ZIP CRC; "
    "image/mask shape, binary and nonempty mask, finite valid image slices"
)


def _read_validated_audit(audit: Path, data_root: Path):
    try:
        with audit.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            "Missing or invalid full-validation audit {}. Use a new manifest path "
            "to validate this data root.".format(audit)
        ) from error
    valid = (
        isinstance(report, dict)
        and report.get("data_root") == str(data_root)
        and report.get("validation_version") == VALIDATION_VERSION
        and report.get("validation_scope") == VALIDATION_SCOPE
        and type(report.get("total")) is int
        and type(report.get("kept")) is int
        and 0 < report["kept"] <= report["total"]
        and isinstance(report.get("rejected"), list)
        and len(report["rejected"]) == report["total"] - report["kept"]
    )
    if valid:
        valid = all(
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and Path(entry["path"]).is_absolute()
            and isinstance(entry.get("error"), str)
            and bool(entry["error"])
            and type(entry.get("size")) is int
            and entry["size"] >= 0
            for entry in report["rejected"]
        )
    if not valid:
        raise ValueError(
            "Full-validation audit {} has a data_root, validation version/scope, "
            "or report mismatch. Use a new manifest path to validate this data root.".format(audit)
        )
    return report


def _validate_npz_file(path: Path):
    try:
        metadata = path.stat()
    except OSError as error:
        raise OSError("Cannot stat NPZ sample {}: {}".format(path, error)) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("NPZ sample is not a regular file: {}".format(path))
    try:
        if metadata.st_size == 0:
            raise ValueError("empty file (0 bytes); expected an NPZ archive")
        # The loader fully decodes the required arrays and checks their content.
        # Do not retain arrays in futures: results contain only rejection metadata.
        ShortAxisNPZDataset.load_validated_arrays(path)
    except ValueError as error:
        cause = error.__cause__ or error.__context__
        seen = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause, OSError) and not isinstance(cause, zipfile.BadZipFile):
                raise OSError("Cannot read NPZ sample {}: {}".format(path, cause)) from error
            if isinstance(cause, (MemoryError, RuntimeError, TypeError, AssertionError)):
                raise cause from error
            cause = cause.__cause__ or cause.__context__
        return {"path": str(path), "error": str(error), "size": metadata.st_size}
    return None


def ensure_validated_manifest(
    data_root: PathLike,
    manifest: PathLike,
    progress: Optional[Callable] = None,
    workers: int = 4,
) -> Path:
    """Reuse a fully audited cohort, or validate all discovered required arrays.

    At most twice ``workers`` files are queued, and only the calling thread emits
    progress. Bad data is reported and excluded; storage and unexpected failures
    abort without publishing a manifest. The final manifest is the completion
    marker and is published atomically after its audit report.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    data_root = Path(data_root).expanduser()
    manifest = Path(manifest).expanduser()
    if progress is not None:
        progress("manifest_prepare", current_path=str(manifest), data_root=str(data_root),
                 validation_version=VALIDATION_VERSION, workers=workers)
    data_root = data_root.resolve()
    manifest = manifest.resolve()
    audit = manifest.with_suffix(".audit.json")
    try:
        manifest_stat = manifest.stat()
    except FileNotFoundError:
        manifest_stat = None
    if manifest_stat is not None:
        if not stat.S_ISREG(manifest_stat.st_mode):
            raise ValueError("manifest is not a regular file: {}".format(manifest))
        report = _read_validated_audit(audit, data_root)
        if progress is not None:
            progress("manifest_ready", current_path=str(manifest), audit_path=str(audit),
                     reused=True, total=report["total"], checked=report["total"],
                     kept=report["kept"], rejected=len(report["rejected"]))
        return manifest

    dataset = ShortAxisNPZDataset(data_root, progress=progress)
    files = sorted(dataset.canonical_files)
    kept, rejected = [], []
    remaining = iter(files)
    pending = {}
    checked = 0

    def update(current_path):
        if progress is not None:
            progress("manifest_audit", current_path=str(current_path), total=len(files),
                     checked=checked, kept=len(kept), rejected=len(rejected), workers=workers)

    update(files[0])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            while True:
                while len(pending) < workers * 2:
                    path = next(remaining, None)
                    if path is None:
                        break
                    pending[executor.submit(_validate_npz_file, path)] = path
                if not pending:
                    break
                # StartupProgress's independent heartbeat continues during slow I/O.
                update(next(iter(pending.values())))
                done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    path = pending.pop(future)
                    result = future.result()
                    if result is None:
                        kept.append(str(path))
                    else:
                        rejected.append(result)
                    checked += 1
                    update(path)
        finally:
            for future in pending:
                future.cancel()
    if not kept:
        raise ValueError("no valid NPZ samples were found in {} ({} rejected files)".format(
            data_root, len(rejected)))
    kept.sort()
    rejected.sort(key=lambda entry: entry["path"])
    report = {
        "data_root": str(data_root), "total": len(files), "kept": len(kept),
        "rejected": rejected, "validation_scope": VALIDATION_SCOPE,
        "validation_version": VALIDATION_VERSION,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(manifest.parent),
            prefix=".{}-".format(manifest.name), suffix=".tmp", delete=False,
        ) as handle:
            temporary_manifest = Path(handle.name)
            temporary_paths.append(temporary_manifest)
            handle.write("\n".join(kept) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(audit.parent),
            prefix=".{}-".format(audit.name), suffix=".tmp", delete=False,
        ) as handle:
            temporary_audit = Path(handle.name)
            temporary_paths.append(temporary_audit)
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_audit), str(audit))
        os.replace(str(temporary_manifest), str(manifest))
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if progress is not None:
        progress("manifest_ready", current_path=str(manifest), audit_path=str(audit),
                 reused=False, total=len(files), checked=checked,
                 kept=len(kept), rejected=len(rejected))
    return manifest


def main():
    """Run a CPU-only full audit with periodic progress on standard output."""
    import argparse
    from SaCMR_encoder.startup import StartupProgress

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with StartupProgress() as progress:
        ensure_validated_manifest(args.data_root, args.manifest,
                                  progress=progress.update, workers=args.workers)


if __name__ == "__main__":
    main()
