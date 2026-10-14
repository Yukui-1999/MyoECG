"""Dataset for preprocessed UK Biobank short-axis CMR volumes."""

import os
import time
import zipfile
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


PathLike = Union[str, os.PathLike]


class ShortAxisNPZDataset(Dataset):
    """Load short-axis NPZ volumes and resize their spatial dimensions.

    Files are indexed without opening their arrays. Each sample is validated and
    loaded lazily by :meth:`__getitem__`. ``canonical_files`` stores resolved
    paths in the same order as ``files`` for cohort fingerprints and snapshots.
    An optional ``progress(event, **details)`` callback receives indexing state.
    ``manifest_is_canonical=True`` trusts absolute, normalized paths from a
    previously generated manifest, avoiding per-sample filesystem metadata reads.
    """

    IMAGE_SHAPE = (112, 112, 10, 50)
    MASK_SHAPE = (10,)

    def __init__(
        self,
        data_root: PathLike,
        manifest: Optional[PathLike] = None,
        size: int = 96,
        max_samples: Optional[int] = None,
        progress: Optional[Callable] = None,
        manifest_is_canonical: bool = False,
    ) -> None:
        self.data_root = Path(data_root).expanduser()
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("size must be a positive integer")
        if max_samples is not None and (
            not isinstance(max_samples, int)
            or isinstance(max_samples, bool)
            or max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer or None")
        if manifest_is_canonical and manifest is None:
            raise ValueError("manifest_is_canonical requires a manifest")
        self.size = size

        if progress is not None:
            progress("index_start", current_path=str(self.data_root),
                     phase="manifest" if manifest is not None else "root",
                     files_found=0)
        if manifest is not None:
            files, canonical_files = self._read_manifest(
                Path(manifest).expanduser(), max_samples, progress,
                manifest_is_canonical)
        else:
            files = self._discover(max_samples, progress)
            canonical_files = list(files)

        if not files:
            raise ValueError("no NPZ samples were found")
        self.files: List[Path] = files
        self.canonical_files: List[Path] = canonical_files

    def _read_manifest(
        self, manifest: Path, max_samples: Optional[int] = None,
        progress: Optional[Callable] = None,
        manifest_is_canonical: bool = False,
    ) -> Tuple[List[Path], List[Path]]:
        files = []
        if progress is not None:
            progress("indexing", phase="manifest", current_path=str(manifest),
                     files_found=0)
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                entry = line.strip()
                if not entry:
                    continue
                path = Path(entry)
                if not manifest_is_canonical:
                    path = path.expanduser()
                    if not path.is_absolute():
                        path = self.data_root / path
                if path.suffix.lower() != ".npz":
                    raise ValueError(
                        "manifest entry {} is not an NPZ path: {}".format(
                            line_number, path
                        )
                    )
                if manifest_is_canonical and (
                    not path.is_absolute()
                    or ".." in path.parts
                    or str(path) != entry
                ):
                    raise ValueError(
                        "canonical manifest entry {} must be an absolute, "
                        "lexically normalized NPZ path: {}".format(line_number, entry)
                    )
                if progress is not None:
                    progress("indexing", phase="manifest", current_path=str(path),
                             files_found=len(files), line_number=line_number)
                canonical_path = path if manifest_is_canonical else path.resolve()
                if not manifest_is_canonical and not canonical_path.is_file():
                    raise FileNotFoundError(path)
                files.append((path, canonical_path))
        files.sort(key=lambda pair: pair[0])
        if max_samples is not None:
            files = files[:max_samples]
        if progress is not None:
            progress("index_complete", phase="manifest", current_path=str(manifest),
                     files_found=len(files))
        return [path for path, _ in files], [path for _, path in files]

    def _discover(self, max_samples: Optional[int],
                  progress: Optional[Callable] = None) -> List[Path]:
        if progress is not None:
            progress("indexing", phase="root", current_path=str(self.data_root),
                     root_entries=0, total_subjects=0, subjects_scanned=0,
                     files_found=0)
        # Descendants are accepted only when scandir says they are not symlinks.
        # Resolving their common root once therefore canonicalizes every discovered
        # path without repeating an NFS metadata walk for each sample.
        self.data_root = self.data_root.resolve()
        if not self.data_root.is_dir():
            raise ValueError("data_root is not a directory: {}".format(self.data_root))

        subject_dirs = []
        root_entries = 0
        last_update = time.monotonic()
        if progress is not None:
            progress("indexing", phase="root", current_path=str(self.data_root),
                     root_entries=0, total_subjects=0, subjects_scanned=0,
                     files_found=0)
        with os.scandir(str(self.data_root)) as entries:
            for entry in entries:
                root_entries += 1
                if entry.is_dir(follow_symlinks=False):
                    subject_dirs.append(Path(entry.path))
                if progress is not None and time.monotonic() - last_update >= 1.0:
                    progress("indexing", phase="root", current_path=str(self.data_root),
                             root_entries=root_entries, total_subjects=len(subject_dirs),
                             subjects_scanned=0, files_found=0)
                    last_update = time.monotonic()
        subject_dirs.sort()
        if progress is not None:
            progress("indexing", phase="root", current_path=str(self.data_root),
                     root_entries=root_entries, total_subjects=len(subject_dirs),
                     subjects_scanned=0, files_found=0)

        files = []
        subjects_scanned = 0
        current_path = self.data_root
        for subject_dir in subject_dirs:
            current_path = subject_dir
            # The callback updates heartbeat state before potentially blocking I/O;
            # it decides how often to print and is never retained on the dataset.
            if progress is not None:
                progress("indexing", phase="subjects", current_path=str(subject_dir),
                         subjects_scanned=subjects_scanned,
                         total_subjects=len(subject_dirs), files_found=len(files))
            with os.scandir(str(subject_dir)) as entries:
                subject_files = sorted(
                    Path(entry.path)
                    for entry in entries
                    if entry.name.lower().endswith(".npz")
                    and entry.is_file(follow_symlinks=False)
                )
            subjects_scanned += 1
            if max_samples is not None:
                subject_files = subject_files[:max_samples - len(files)]
            files.extend(subject_files)
            if progress is not None:
                progress("indexing", phase="subjects", current_path=str(subject_dir),
                         subjects_scanned=subjects_scanned,
                         total_subjects=len(subject_dirs), files_found=len(files))
            if max_samples is not None and len(files) >= max_samples:
                break
        if progress is not None:
            progress("index_complete", phase="subjects", current_path=str(current_path),
                     subjects_scanned=subjects_scanned,
                     total_subjects=len(subject_dirs), files_found=len(files))
        return files

    def __len__(self) -> int:
        return len(self.files)

    @classmethod
    def load_validated_arrays(cls, path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
        """Decode and validate required arrays without resizing or tensor conversion.

        Return the image in its stored dtype and a boolean slice mask. Invalid
        slices are left unchanged; :meth:`__getitem__` masks them before resizing.
        Reading both required NPZ members also verifies their ZIP checksums.
        """
        path = Path(path)
        try:
            with path.open("rb") as handle:
                header = handle.read(4)
                if not header:
                    raise ValueError("empty file (0 bytes); expected an NPZ archive")
                if header not in (b"PK\x03\x04", b"PK\x05\x06"):
                    raise ValueError(
                        "invalid NPZ ZIP header: {}".format(header.hex())
                    )
                handle.seek(0)
                sample = np.load(handle, allow_pickle=False)
                if not isinstance(sample, np.lib.npyio.NpzFile):
                    raise ValueError("expected an NPZ archive, got {}".format(
                        type(sample).__name__))
                with sample:
                    missing = {"image", "slice_valid_mask"}.difference(sample.files)
                    if missing:
                        raise ValueError(
                            "{} is missing required field(s): {}".format(
                                path, ", ".join(sorted(missing))
                            )
                        )
                    image = sample["image"]
                    mask = sample["slice_valid_mask"]
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile,
                zlib.error, RuntimeError) as error:
            # NumPy and ZIP codecs raise different exceptions while opening and
            # lazily decoding arrays. Keep their cause and the exact sample path.
            raise ValueError(
                "Failed to load NPZ sample {}: {}: {}. "
                "Check or regenerate this file before training.".format(
                    path, type(error).__name__, error
                )
            ) from error

        if image.shape != cls.IMAGE_SHAPE:
            raise ValueError(
                "{} image shape must be {}, got {}".format(
                    path, cls.IMAGE_SHAPE, image.shape
                )
            )
        if mask.shape != cls.MASK_SHAPE:
            raise ValueError(
                "{} slice_valid_mask shape must be {}, got {}".format(
                    path, cls.MASK_SHAPE, mask.shape
                )
            )
        if not np.all(np.isin(mask, (0, 1))):
            raise ValueError("{} slice_valid_mask must be binary".format(path))

        valid_mask = mask.astype(bool, copy=False)
        if not valid_mask.any():
            raise ValueError("{} has no valid slices".format(path))
        if not np.isfinite(image[:, :, valid_mask, :]).all():
            raise ValueError("{} contains nonfinite values in valid slices".format(path))

        return image, valid_mask

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.files[index]
        image, valid_mask = self.load_validated_arrays(path)

        # Copy into float32 before mutation: np.load arrays may be read-only, and
        # invalid slices can contain duplicated signal or nonfinite placeholders.
        image = image.astype(np.float32, copy=True)
        image[:, :, ~valid_mask, :] = 0.0
        image_tensor = torch.from_numpy(image).permute(2, 3, 0, 1).contiguous()
        if self.size != self.IMAGE_SHAPE[0]:
            image_tensor = F.interpolate(
                image_tensor,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "image": image_tensor,
            "slice_valid_mask": torch.from_numpy(valid_mask.copy()),
            "case_id": path.parent.name,
            "path": str(path),
        }
