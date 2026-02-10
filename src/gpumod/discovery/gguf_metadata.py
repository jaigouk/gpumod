"""GGUF file metadata fetcher from HuggingFace.

Fetches GGUF file listings and sizes from HuggingFace repos
without downloading the actual files.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# VRAM overhead factor (10% for runtime buffers, KV cache scratch)
_VRAM_OVERHEAD_FACTOR = 1.1


class RepoNotFoundError(Exception):
    """Raised when the HuggingFace repo is not found."""


@dataclass(frozen=True)
class GGUFFile:
    """Immutable metadata for a GGUF file.

    Attributes:
        filename: Name of the GGUF file.
        size_bytes: File size in bytes.
        quant_type: Detected quantization type (e.g., "Q4_K_M") or None.
        estimated_vram_mb: Estimated VRAM requirement in MB.
        is_split: Whether this is part of a split model.
        split_parts: Total number of parts if split, else 1.
    """

    filename: str
    size_bytes: int
    quant_type: str | None
    estimated_vram_mb: int
    is_split: bool
    split_parts: int


class GGUFMetadataFetcher:
    """Fetches GGUF file metadata from HuggingFace repos.

    Lists GGUF files in a repo, gets their sizes, parses quantization
    types from filenames, and estimates VRAM requirements.

    Example:
        >>> fetcher = GGUFMetadataFetcher()
        >>> files = await fetcher.list_gguf_files("unsloth/Qwen3-Coder-Next-GGUF")
        >>> for f in files:
        ...     print(f"{f.filename}: {f.quant_type}, {f.estimated_vram_mb} MB")
    """

    # Quantization patterns ordered by specificity (longest first)
    # These match llama.cpp quantization naming conventions
    _QUANT_PATTERNS: tuple[str, ...] = (
        # Unsloth dynamic quants (match these first)
        r"UD-Q[2-8]_K_XL",
        r"UD-Q[2-8]_K_[SMLX]+",
        # Extended quants
        r"Q[2-8]_K_XL",
        r"Q8_K_XL",
        # Standard llama.cpp quants
        r"Q[2-8]_K_[SM]",
        r"Q[2-8]_K",
        r"Q[2-8]_[01]",
        # IQuants
        r"IQ[1-4]_[SMLX]+",
        r"IQ[1-4]_NL",
    )

    # Compiled regex for quant detection (case insensitive)
    _QUANT_REGEX = re.compile(
        "|".join(f"({p})" for p in _QUANT_PATTERNS),
        re.IGNORECASE,
    )

    # Pattern for split files: model-00001-of-00003.gguf
    _SPLIT_PATTERN = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

    def __init__(self) -> None:
        """Initialize the fetcher."""

    async def list_gguf_files(  # noqa: C901, PLR0912, PLR0915
        self, repo_id: str
    ) -> list[GGUFFile]:
        """List all GGUF files in a HuggingFace repo with metadata.

        Args:
            repo_id: HuggingFace repo ID (e.g., "unsloth/Qwen3-Coder-Next-GGUF").

        Returns:
            List of GGUFFile sorted by size (smallest first).

        Raises:
            RepoNotFoundError: If the repo doesn't exist.
        """
        import time

        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError

        api = HfApi()

        # Get repo info with file sizes in single request (faster than per-file)
        start = time.monotonic()
        try:
            repo_info = await asyncio.to_thread(api.repo_info, repo_id, files_metadata=True)
            files_with_sizes = {f.rfilename: f.size for f in (repo_info.siblings or []) if f.size}
            files = list(files_with_sizes.keys())
            elapsed = time.monotonic() - start
            logger.debug("Got %d files from repo_info in %.2fs", len(files), elapsed)
        except RepositoryNotFoundError as exc:
            raise RepoNotFoundError(f"Repo not found: {repo_id}") from exc
        except Exception as exc:
            # Fallback to list_repo_files if repo_info fails
            logger.debug("repo_info failed, falling back to list_repo_files: %s", exc)
            files_with_sizes = {}
            try:
                files = await asyncio.to_thread(api.list_repo_files, repo_id)
            except RepositoryNotFoundError as exc2:
                raise RepoNotFoundError(f"Repo not found: {repo_id}") from exc2
            except Exception as exc2:
                if "not found" in str(exc2).lower():
                    raise RepoNotFoundError(f"Repo not found: {repo_id}") from exc2
                raise

        # Filter to GGUF files, excluding non-model files
        # imatrix files are calibration data, not actual models
        gguf_files = [
            f for f in files if f.lower().endswith(".gguf") and "imatrix" not in f.lower()
        ]

        if not gguf_files:
            return []

        # Group split files
        split_groups: dict[str, list[str]] = {}
        single_files: list[str] = []

        for filename in gguf_files:
            match = self._SPLIT_PATTERN.search(filename)
            if match:
                # Extract base name (before the split suffix)
                base = self._SPLIT_PATTERN.sub("", filename)
                if base not in split_groups:
                    split_groups[base] = []
                split_groups[base].append(filename)
            else:
                single_files.append(filename)

        # Fetch metadata for each file
        results: list[GGUFFile] = []

        # Helper to get file size (use cached if available)
        async def get_size(filename: str) -> int:
            if filename in files_with_sizes:
                return files_with_sizes[filename]
            return await self._get_file_size(api, repo_id, filename)

        # Process single files
        for filename in single_files:
            size = await get_size(filename)
            quant = self._parse_quant_type(filename)
            vram = self._estimate_vram(size)

            results.append(
                GGUFFile(
                    filename=filename,
                    size_bytes=size,
                    quant_type=quant,
                    estimated_vram_mb=vram,
                    is_split=False,
                    split_parts=1,
                )
            )

        # Process split file groups
        for parts in split_groups.values():
            total_size = 0
            for part in parts:
                size = await get_size(part)
                total_size += size

            # Use first part's filename for display
            first_part = sorted(parts)[0]
            quant = self._parse_quant_type(first_part)
            vram = self._estimate_vram(total_size)

            results.append(
                GGUFFile(
                    filename=first_part,
                    size_bytes=total_size,
                    quant_type=quant,
                    estimated_vram_mb=vram,
                    is_split=True,
                    split_parts=len(parts),
                )
            )

        # Sort by size (smallest first for discoverability)
        results.sort(key=lambda f: f.size_bytes)

        return results

    async def _get_file_size(
        self,
        api: object,
        repo_id: str,
        filename: str,
    ) -> int:
        """Get file size from HuggingFace metadata.

        Args:
            api: HfApi instance.
            repo_id: Repository ID.
            filename: File path in repo.

        Returns:
            File size in bytes, or 0 if unavailable.
        """
        from huggingface_hub import get_hf_file_metadata

        try:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            metadata = await asyncio.to_thread(get_hf_file_metadata, url)
            return metadata.size or 0
        except Exception as exc:
            logger.warning("Failed to get size for %s: %s", filename, exc)
            return 0

    def _parse_quant_type(self, filename: str) -> str | None:
        """Parse quantization type from filename.

        Args:
            filename: GGUF filename.

        Returns:
            Quantization type (uppercase) or None if not detected.
        """
        match = self._QUANT_REGEX.search(filename)
        if match:
            # Return the matched group, uppercased
            return match.group(0).upper()
        return None

    def _estimate_vram(self, size_bytes: int) -> int:
        """Estimate VRAM from file size.

        Uses file_size * 1.1 formula (10% overhead for runtime buffers).

        Args:
            size_bytes: File size in bytes.

        Returns:
            Estimated VRAM in megabytes.
        """
        if size_bytes == 0:
            return 0
        size_mb = size_bytes / (1024 * 1024)
        return math.ceil(size_mb * _VRAM_OVERHEAD_FACTOR)
