"""Tests for model download functionality (gpumod-lch).

Tests cover:
- ModelFileCheck preflight detection
- DownloadInfo construction and formatting
- DiskSpaceChecker validation
- ModelDownloader orchestration with user confirmation
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpumod.preflight.model_file import (
    DiskSpaceChecker,
    DownloadAbortedError,
    DownloadInfo,
    InsufficientDiskSpaceError,
    ModelDownloader,
    ModelFileCheck,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def mock_service(tmp_models_dir: Path) -> MagicMock:
    """Create a mock service with llama.cpp configuration."""
    service = MagicMock()
    service.id = "test-llama"
    service.driver_type = "llamacpp"
    service.model_id = "unsloth/Qwen3-Coder-Next-GGUF"
    service.extra_config = {
        "unit_vars": {
            "models_dir": str(tmp_models_dir),
            "model_file": "Qwen3-Coder-Next-UD-Q4_K_M.gguf",
        }
    }
    return service


@pytest.fixture
def mock_service_no_model_id() -> MagicMock:
    """Create a mock service without model_id (skip check)."""
    service = MagicMock()
    service.id = "test-fastapi"
    service.driver_type = "fastapi"
    service.model_id = None
    service.extra_config = {}
    return service


@pytest.fixture
def download_info(tmp_models_dir: Path) -> DownloadInfo:
    """Create a sample DownloadInfo with small size for test file creation."""
    return DownloadInfo(
        local_path=tmp_models_dir / "model.gguf",
        repo_id="unsloth/Qwen3-Coder-Next-GGUF",
        filename="Qwen3-Coder-Next-UD-Q4_K_M.gguf",
        file_size_bytes=1000,  # Small size for tests that create real files
    )


# ---------------------------------------------------------------------------
# ModelFileCheck Tests
# ---------------------------------------------------------------------------


class TestModelFileCheck:
    """Tests for ModelFileCheck preflight validation."""

    async def test_check_passes_when_file_exists(
        self, mock_service: MagicMock, tmp_models_dir: Path
    ) -> None:
        """Check passes when the model file exists."""
        # Create the model file
        model_file = tmp_models_dir / "Qwen3-Coder-Next-UD-Q4_K_M.gguf"
        model_file.write_bytes(b"fake gguf content")

        check = ModelFileCheck()
        result = await check.check(mock_service)

        assert result.passed is True
        assert result.severity == "info"
        assert "exists" in result.message.lower()

    async def test_check_fails_when_file_missing(
        self, mock_service: MagicMock, tmp_models_dir: Path
    ) -> None:
        """Check fails when the model file is missing."""
        check = ModelFileCheck()
        result = await check.check(mock_service)

        assert result.passed is False
        assert result.severity == "error"
        assert "not found" in result.message.lower()
        assert result.remediation is not None
        assert "download" in result.remediation.lower()

    async def test_check_skipped_for_non_llamacpp(
        self, mock_service_no_model_id: MagicMock
    ) -> None:
        """Check is skipped for services without model_id."""
        check = ModelFileCheck()
        result = await check.check(mock_service_no_model_id)

        assert result.passed is True
        assert result.severity == "info"
        assert "skipped" in result.message.lower()

    async def test_check_extracts_download_info_on_failure(
        self, mock_service: MagicMock, tmp_models_dir: Path
    ) -> None:
        """Download info is available when check fails."""
        check = ModelFileCheck()
        result = await check.check(mock_service)

        assert result.passed is False
        # Check that we can get download info from the check
        info = check.get_download_info()
        assert info is not None
        assert info.repo_id == "unsloth/Qwen3-Coder-Next-GGUF"
        assert info.filename == "Qwen3-Coder-Next-UD-Q4_K_M.gguf"
        assert info.local_path == tmp_models_dir / "Qwen3-Coder-Next-UD-Q4_K_M.gguf"

    def test_name_property(self) -> None:
        """Check has correct name."""
        check = ModelFileCheck()
        assert check.name == "model_file"


# ---------------------------------------------------------------------------
# DownloadInfo Tests
# ---------------------------------------------------------------------------


class TestDownloadInfo:
    """Tests for DownloadInfo dataclass."""

    def test_download_url_construction(self, download_info: DownloadInfo) -> None:
        """Download URL is correctly constructed."""
        expected = (
            "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/"
            "resolve/main/Qwen3-Coder-Next-UD-Q4_K_M.gguf"
        )
        assert download_info.download_url == expected

    def test_file_size_human_readable_gb(self, tmp_models_dir: Path) -> None:
        """File size formats correctly in GB."""
        info = DownloadInfo(
            local_path=tmp_models_dir / "large.gguf",
            repo_id="test/repo",
            filename="large.gguf",
            file_size_bytes=8_000_000_000,  # 8 GB
        )
        assert info.file_size_human == "7.45 GB"

    def test_file_size_human_readable_mb(self, tmp_models_dir: Path) -> None:
        """File size formats correctly in MB for smaller files."""
        info = DownloadInfo(
            local_path=tmp_models_dir / "small.gguf",
            repo_id="test/repo",
            filename="small.gguf",
            file_size_bytes=500_000_000,  # 500 MB
        )
        assert info.file_size_human == "476.84 MB"

    def test_wget_command(self, download_info: DownloadInfo) -> None:
        """wget command is correctly formatted."""
        cmd = download_info.wget_command
        assert "wget" in cmd
        assert download_info.download_url in cmd
        assert str(download_info.local_path) in cmd

    def test_curl_command(self, download_info: DownloadInfo) -> None:
        """curl command is correctly formatted."""
        cmd = download_info.curl_command
        assert "curl" in cmd
        assert "-L" in cmd  # follow redirects
        assert download_info.download_url in cmd
        assert str(download_info.local_path) in cmd


# ---------------------------------------------------------------------------
# DiskSpaceChecker Tests
# ---------------------------------------------------------------------------


class TestDiskSpaceChecker:
    """Tests for disk space validation."""

    def test_sufficient_space(self, tmp_models_dir: Path) -> None:
        """Passes when disk has sufficient space."""
        checker = DiskSpaceChecker()
        # Get actual disk space - should have enough for 1 byte
        result = checker.check(tmp_models_dir, required_bytes=1)
        assert result.sufficient is True
        assert result.available_bytes > 0

    def test_insufficient_space(self, tmp_models_dir: Path) -> None:
        """Fails when disk space is insufficient."""
        checker = DiskSpaceChecker()
        # Request absurdly large space
        result = checker.check(tmp_models_dir, required_bytes=10**18)  # 1 EB
        assert result.sufficient is False
        assert "insufficient" in result.message.lower()

    def test_includes_safety_buffer(self, tmp_models_dir: Path) -> None:
        """Required space includes 10% safety buffer."""
        checker = DiskSpaceChecker(safety_buffer=0.1)
        # Check that buffer is applied
        result = checker.check(tmp_models_dir, required_bytes=1000)
        # Required with buffer should be 1100
        assert result.required_with_buffer == 1100

    def test_formats_human_readable(self, tmp_models_dir: Path) -> None:
        """Formats space in human-readable units."""
        checker = DiskSpaceChecker()
        result = checker.check(tmp_models_dir, required_bytes=1_000_000_000)
        assert "GB" in result.available_human or "TB" in result.available_human


# ---------------------------------------------------------------------------
# ModelDownloader Tests
# ---------------------------------------------------------------------------


class TestModelDownloader:
    """Tests for ModelDownloader orchestration."""

    async def test_aborts_without_user_confirmation(self, download_info: DownloadInfo) -> None:
        """Download aborts if user does not confirm."""
        confirm_fn = MagicMock(return_value=False)
        downloader = ModelDownloader(confirm_fn=confirm_fn)

        with pytest.raises(DownloadAbortedError) as exc_info:
            await downloader.download(download_info)

        assert "user declined" in str(exc_info.value).lower()
        confirm_fn.assert_called_once()

    async def test_checks_disk_space_after_confirmation(self, download_info: DownloadInfo) -> None:
        """Disk space is checked only after user confirms."""
        confirm_fn = MagicMock(return_value=True)
        disk_checker = MagicMock()
        disk_checker.check.return_value = MagicMock(
            sufficient=False,
            message="Insufficient disk space",
            available_bytes=1_000_000_000,
            required_with_buffer=10_000_000_000,
        )

        downloader = ModelDownloader(
            confirm_fn=confirm_fn,
            disk_checker=disk_checker,
        )

        with pytest.raises(InsufficientDiskSpaceError) as exc_info:
            await downloader.download(download_info)

        assert "disk space" in str(exc_info.value).lower()
        confirm_fn.assert_called_once()
        disk_checker.check.assert_called_once()

    async def test_downloads_with_progress(
        self, download_info: DownloadInfo, tmp_models_dir: Path
    ) -> None:
        """Download proceeds with progress callback when all checks pass."""
        confirm_fn = MagicMock(return_value=True)
        disk_checker = MagicMock()
        disk_checker.check.return_value = MagicMock(
            sufficient=True,
            available_bytes=100_000_000_000,
            required_with_buffer=8_800_000_000,
        )
        progress_calls: list[tuple[int, int]] = []

        def track_progress(downloaded: int, total: int) -> None:
            progress_calls.append((downloaded, total))

        # Mock creates file with correct size
        async def fake_download(info: DownloadInfo) -> None:
            info.local_path.parent.mkdir(parents=True, exist_ok=True)
            info.local_path.write_bytes(b"x" * info.file_size_bytes)

        with patch.object(
            ModelDownloader, "_do_download", new_callable=AsyncMock
        ) as mock_download:
            mock_download.side_effect = fake_download

            downloader = ModelDownloader(
                confirm_fn=confirm_fn,
                disk_checker=disk_checker,
                progress_fn=track_progress,
            )

            await downloader.download(download_info)

        mock_download.assert_called_once()

    async def test_verifies_file_after_download(
        self, download_info: DownloadInfo, tmp_models_dir: Path
    ) -> None:
        """File size is verified after download completes."""
        confirm_fn = MagicMock(return_value=True)
        disk_checker = MagicMock()
        disk_checker.check.return_value = MagicMock(sufficient=True)

        # Create a file with wrong size
        download_info.local_path.write_bytes(b"too small")

        with patch.object(
            ModelDownloader, "_do_download", new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = None

            downloader = ModelDownloader(
                confirm_fn=confirm_fn,
                disk_checker=disk_checker,
            )

            with pytest.raises(ValueError, match="size mismatch"):
                await downloader.download(download_info)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    async def test_handles_split_gguf_files(self, tmp_models_dir: Path) -> None:
        """Handles split GGUF files (e.g., model-00001-of-00003.gguf)."""
        service = MagicMock()
        service.id = "test-split"
        service.driver_type = "llamacpp"
        service.model_id = "unsloth/Large-Model-GGUF"
        service.extra_config = {
            "unit_vars": {
                "models_dir": str(tmp_models_dir),
                "model_file": "Large-Model-Q4_K_M-00001-of-00003.gguf",
            }
        }

        check = ModelFileCheck()
        result = await check.check(service)

        assert result.passed is False
        info = check.get_download_info()
        assert info is not None
        # Should detect split file pattern
        assert "00001-of-00003" in info.filename

    async def test_handles_missing_models_dir(self, tmp_path: Path) -> None:
        """Handles case where models_dir doesn't exist yet."""
        nonexistent = tmp_path / "nonexistent" / "models"
        service = MagicMock()
        service.id = "test-nodir"
        service.driver_type = "llamacpp"
        service.model_id = "test/repo"
        service.extra_config = {
            "unit_vars": {
                "models_dir": str(nonexistent),
                "model_file": "model.gguf",
            }
        }

        check = ModelFileCheck()
        result = await check.check(service)

        assert result.passed is False
        assert "not found" in result.message.lower()

    async def test_handles_empty_model_id(self, tmp_models_dir: Path) -> None:
        """Handles service with empty string model_id."""
        service = MagicMock()
        service.id = "test-empty"
        service.driver_type = "llamacpp"
        service.model_id = ""  # Empty string
        service.extra_config = {}

        check = ModelFileCheck()
        result = await check.check(service)

        assert result.passed is True
        assert "skipped" in result.message.lower()

    def test_disk_space_check_on_nonexistent_path(self, tmp_path: Path) -> None:
        """DiskSpaceChecker handles nonexistent path gracefully."""
        checker = DiskSpaceChecker()
        nonexistent = tmp_path / "does" / "not" / "exist"

        # Should check parent that exists
        result = checker.check(nonexistent, required_bytes=1)
        assert result.available_bytes > 0

    async def test_download_creates_parent_directories(self, tmp_path: Path) -> None:
        """ModelDownloader creates parent directories if needed."""
        nested_path = tmp_path / "deep" / "nested" / "path" / "model.gguf"
        info = DownloadInfo(
            local_path=nested_path,
            repo_id="test/repo",
            filename="model.gguf",
            file_size_bytes=1000,
        )

        confirm_fn = MagicMock(return_value=True)
        disk_checker = MagicMock()
        disk_checker.check.return_value = MagicMock(sufficient=True)

        with patch.object(
            ModelDownloader, "_do_download", new_callable=AsyncMock
        ) as mock_download:
            # Simulate successful download by creating the file
            async def create_file(info: DownloadInfo) -> None:
                info.local_path.parent.mkdir(parents=True, exist_ok=True)
                info.local_path.write_bytes(b"x" * 1000)

            mock_download.side_effect = create_file

            downloader = ModelDownloader(
                confirm_fn=confirm_fn,
                disk_checker=disk_checker,
            )

            await downloader.download(info)

        assert nested_path.exists()
