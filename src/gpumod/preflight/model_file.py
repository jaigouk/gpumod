"""Model file preflight check and download support (gpumod-lch).

Provides:
- ModelFileCheck: Preflight check for missing GGUF model files
- DownloadInfo: Metadata for downloading a model file
- DiskSpaceChecker: Validates available disk space before download
- ModelDownloader: Orchestrates download with user confirmation

Key Requirement: Downloads are NEVER automatic. User MUST confirm.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gpumod.preflight.base import CheckResult

if TYPE_CHECKING:
    from gpumod.models import Service


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DownloadAbortedError(Exception):
    """Raised when user declines the download."""


class InsufficientDiskSpaceError(Exception):
    """Raised when disk space is insufficient for download."""


# ---------------------------------------------------------------------------
# DownloadInfo
# ---------------------------------------------------------------------------


@dataclass
class DownloadInfo:
    """Metadata for downloading a model file.

    Attributes:
        local_path: Full path where file will be saved.
        repo_id: HuggingFace repository ID.
        filename: Name of the file in the repository.
        file_size_bytes: File size in bytes (0 if unknown).
    """

    local_path: Path
    repo_id: str
    filename: str
    file_size_bytes: int = 0

    @property
    def download_url(self) -> str:
        """Direct download URL from HuggingFace."""
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"

    @property
    def file_size_human(self) -> str:
        """Human-readable file size."""
        return _format_bytes(self.file_size_bytes)

    @property
    def wget_command(self) -> str:
        """wget command for manual download."""
        return f'wget -c "{self.download_url}" -O "{self.local_path}"'

    @property
    def curl_command(self) -> str:
        """curl command for manual download."""
        return f'curl -L -C - "{self.download_url}" -o "{self.local_path}"'


# ---------------------------------------------------------------------------
# DiskSpaceChecker
# ---------------------------------------------------------------------------


@dataclass
class DiskSpaceResult:
    """Result of a disk space check.

    Attributes:
        sufficient: Whether there is enough space.
        available_bytes: Available disk space in bytes.
        required_with_buffer: Required space including safety buffer.
        message: Human-readable message.
    """

    sufficient: bool
    available_bytes: int
    required_with_buffer: int
    message: str

    @property
    def available_human(self) -> str:
        """Human-readable available space."""
        return _format_bytes(self.available_bytes)

    @property
    def required_human(self) -> str:
        """Human-readable required space."""
        return _format_bytes(self.required_with_buffer)


class DiskSpaceChecker:
    """Checks available disk space before download.

    Parameters:
        safety_buffer: Extra space required as fraction (default 0.1 = 10%).
    """

    def __init__(self, safety_buffer: float = 0.1) -> None:
        self._safety_buffer = safety_buffer

    def check(self, path: Path, required_bytes: int) -> DiskSpaceResult:
        """Check if path has sufficient disk space.

        If path doesn't exist, checks the nearest existing parent directory.

        Parameters:
            path: Target path (file or directory).
            required_bytes: Required space in bytes.

        Returns:
            DiskSpaceResult with check outcome.
        """
        # Find existing directory to check
        check_path = path
        while not check_path.exists():
            parent = check_path.parent
            if parent == check_path:  # Reached root
                break
            check_path = parent

        # Get disk usage
        try:
            usage = shutil.disk_usage(check_path)
            available = usage.free
        except OSError:
            available = 0

        # Calculate required with buffer
        required_with_buffer = int(required_bytes * (1 + self._safety_buffer))

        sufficient = available >= required_with_buffer
        if sufficient:
            message = f"Sufficient disk space: {_format_bytes(available)} available"
        else:
            message = (
                f"Insufficient disk space. "
                f"Need {_format_bytes(required_with_buffer)}, "
                f"only {_format_bytes(available)} available."
            )

        return DiskSpaceResult(
            sufficient=sufficient,
            available_bytes=available,
            required_with_buffer=required_with_buffer,
            message=message,
        )


# ---------------------------------------------------------------------------
# ModelFileCheck (PreflightCheck)
# ---------------------------------------------------------------------------


class ModelFileCheck:
    """Preflight check that verifies model file exists.

    For llama.cpp services, checks if the GGUF file exists at the
    expected path. If missing, captures download info for remediation.

    Usage:
        check = ModelFileCheck()
        result = await check.check(service)
        if not result.passed:
            info = check.get_download_info()
            # Offer to download
    """

    def __init__(self) -> None:
        self._last_download_info: DownloadInfo | None = None

    @property
    def name(self) -> str:
        """Return check name."""
        return "model_file"

    async def check(self, service: Service) -> CheckResult:
        """Check if model file exists for the service.

        Parameters:
            service: Service to validate.

        Returns:
            CheckResult indicating pass/fail with remediation hints.
        """
        self._last_download_info = None

        # Skip for services without model_id
        if not service.model_id:
            return CheckResult(
                passed=True,
                severity="info",
                message="Model file check skipped (no model_id configured)",
            )

        # Extract model path from extra_config
        unit_vars = service.extra_config.get("unit_vars", {})
        if not isinstance(unit_vars, dict):
            return CheckResult(
                passed=True,
                severity="info",
                message="Model file check skipped (no unit_vars)",
            )

        models_dir = unit_vars.get("models_dir")
        model_file = unit_vars.get("model_file")

        if not models_dir or not model_file:
            return CheckResult(
                passed=True,
                severity="info",
                message="Model file check skipped (no models_dir or model_file)",
            )

        # Check if file exists
        model_path = Path(models_dir) / model_file
        if model_path.exists():
            return CheckResult(
                passed=True,
                severity="info",
                message=f"Model file exists: {model_path}",
            )

        # File missing - create download info
        self._last_download_info = DownloadInfo(
            local_path=model_path,
            repo_id=service.model_id,
            filename=model_file,
            file_size_bytes=0,  # Will be fetched later if needed
        )

        return CheckResult(
            passed=False,
            severity="error",
            message=f"Model file not found: {model_path}",
            remediation=(
                f"Download the model file from HuggingFace:\n"
                f"  {self._last_download_info.wget_command}"
            ),
        )

    def get_download_info(self) -> DownloadInfo | None:
        """Get download info from last failed check.

        Returns:
            DownloadInfo if last check found a missing file, else None.
        """
        return self._last_download_info


# ---------------------------------------------------------------------------
# ModelDownloader
# ---------------------------------------------------------------------------


# Type alias for confirmation callback
ConfirmFn = Callable[[DownloadInfo], bool]
ProgressFn = Callable[[int, int], None]


class ModelDownloader:
    """Orchestrates model file download with user confirmation.

    Key Requirement: Downloads are NEVER automatic. User MUST confirm.

    Flow:
    1. Display download info to user
    2. Prompt for confirmation (default NO)
    3. Check disk space (after confirmation)
    4. Download with progress
    5. Verify file size

    Parameters:
        confirm_fn: Callback to get user confirmation. Must return True to proceed.
        disk_checker: DiskSpaceChecker instance (default: create new).
        progress_fn: Optional callback for download progress (downloaded, total).
    """

    def __init__(
        self,
        confirm_fn: ConfirmFn,
        disk_checker: DiskSpaceChecker | None = None,
        progress_fn: ProgressFn | None = None,
    ) -> None:
        self._confirm_fn = confirm_fn
        self._disk_checker = disk_checker or DiskSpaceChecker()
        self._progress_fn = progress_fn

    async def download(self, info: DownloadInfo) -> None:
        """Download the model file with user confirmation.

        Parameters:
            info: DownloadInfo with file metadata.

        Raises:
            DownloadAbortedError: If user declines download.
            InsufficientDiskSpaceError: If disk space is insufficient.
            ValueError: If downloaded file size doesn't match.
        """
        # Step 1: Get user confirmation (NEVER automatic)
        if not self._confirm_fn(info):
            raise DownloadAbortedError("Download aborted: user declined")

        # Step 2: Check disk space AFTER confirmation
        space_result = self._disk_checker.check(
            info.local_path.parent,
            info.file_size_bytes,
        )
        if not space_result.sufficient:
            raise InsufficientDiskSpaceError(
                f"Insufficient disk space for download. {space_result.message}"
            )

        # Step 3: Create parent directories if needed
        info.local_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 4: Download the file
        await self._do_download(info)

        # Step 5: Verify file size
        if info.file_size_bytes > 0:
            actual_size = info.local_path.stat().st_size
            if actual_size != info.file_size_bytes:
                raise ValueError(
                    f"Downloaded file size mismatch: "
                    f"expected {info.file_size_bytes}, got {actual_size}"
                )

    async def _do_download(self, info: DownloadInfo) -> None:
        """Perform the actual download with progress.

        This method can be overridden for testing or custom download logic.

        Parameters:
            info: DownloadInfo with URL and local path.
        """
        import asyncio
        from functools import partial

        from huggingface_hub import hf_hub_download

        # Use huggingface_hub for authenticated/resumable downloads
        download_fn = partial(
            hf_hub_download,
            repo_id=info.repo_id,
            filename=info.filename,
            local_dir=str(info.local_path.parent),
            local_dir_use_symlinks=False,
        )
        await asyncio.to_thread(download_fn)

        # Move from HF cache structure to expected location if needed
        downloaded = info.local_path.parent / info.filename
        if downloaded != info.local_path and downloaded.exists():
            downloaded.rename(info.local_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"
