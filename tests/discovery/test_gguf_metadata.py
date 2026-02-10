"""Tests for GGUFMetadataFetcher - RED phase first."""

from __future__ import annotations

import pytest

from gpumod.discovery.gguf_metadata import (
    GGUFFile,
    GGUFMetadataFetcher,
    RepoNotFoundError,
)


class TestGGUFFileDataclass:
    """Tests for GGUFFile dataclass."""

    def test_gguf_file_is_frozen(self) -> None:
        """GGUFFile should be immutable (frozen dataclass)."""
        gguf = GGUFFile(
            filename="model-Q4_K_M.gguf",
            size_bytes=21_000_000_000,
            quant_type="Q4_K_M",
            estimated_vram_mb=22000,
            is_split=False,
            split_parts=1,
        )
        with pytest.raises(AttributeError):
            gguf.filename = "changed"  # type: ignore[misc]

    def test_estimated_vram_calculation(self) -> None:
        """estimated_vram_mb should be file_size * 1.1."""
        size_bytes = 20_000_000_000  # 20 GB
        expected_vram = int((size_bytes / (1024 * 1024)) * 1.1)
        gguf = GGUFFile(
            filename="model.gguf",
            size_bytes=size_bytes,
            quant_type=None,
            estimated_vram_mb=expected_vram,
            is_split=False,
            split_parts=1,
        )
        assert gguf.estimated_vram_mb == expected_vram


class TestGGUFMetadataFetcher:
    """Tests for GGUFMetadataFetcher."""

    @pytest.mark.asyncio
    async def test_list_gguf_files_returns_list(
        self,
        mock_hf_repo_files: None,
    ) -> None:
        """list_gguf_files() should return a list of GGUFFile."""
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files("unsloth/Qwen3-Coder-Next-GGUF")
        assert isinstance(files, list)
        assert all(isinstance(f, GGUFFile) for f in files)

    @pytest.mark.asyncio
    async def test_gets_accurate_file_sizes(
        self,
        mock_hf_repo_files_with_sizes: None,
    ) -> None:
        """Should get accurate file sizes from HF metadata."""
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files("unsloth/test-model")
        assert len(files) > 0
        assert all(f.size_bytes > 0 for f in files)

    @pytest.mark.asyncio
    async def test_parses_q4_k_m_from_filename(self) -> None:
        """Should parse Q4_K_M from 'model-Q4_K_M.gguf'."""
        fetcher = GGUFMetadataFetcher()
        quant = fetcher._parse_quant_type("Qwen3-Coder-30B-Q4_K_M.gguf")
        assert quant == "Q4_K_M"

    @pytest.mark.asyncio
    async def test_parses_ud_q4_k_xl_from_filename(self) -> None:
        """Should parse UD-Q4_K_XL from 'Model-UD-Q4_K_XL.gguf'."""
        fetcher = GGUFMetadataFetcher()
        quant = fetcher._parse_quant_type("Nemotron-UD-Q4_K_XL.gguf")
        assert quant == "UD-Q4_K_XL"

    @pytest.mark.asyncio
    async def test_parses_q2_k_xl_from_filename(self) -> None:
        """Should parse Q2_K_XL from filename."""
        fetcher = GGUFMetadataFetcher()
        quant = fetcher._parse_quant_type("Model-Q2_K_XL.gguf")
        assert quant == "Q2_K_XL"

    @pytest.mark.asyncio
    async def test_handles_split_files(
        self,
        mock_hf_repo_split_files: None,
    ) -> None:
        """Should handle split GGUF files and sum all parts."""
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files("unsloth/large-model")
        # Should consolidate split files
        split_files = [f for f in files if f.is_split]
        if split_files:
            assert split_files[0].split_parts > 1

    @pytest.mark.asyncio
    async def test_vram_estimate_formula(self) -> None:
        """VRAM estimate should be file_size_bytes * 1.1 / 1024 / 1024."""
        import math

        fetcher = GGUFMetadataFetcher()
        size_bytes = 21_000_000_000
        vram = fetcher._estimate_vram(size_bytes)
        # Formula uses math.ceil for rounding up
        expected = math.ceil((size_bytes / (1024 * 1024)) * 1.1)
        assert vram == expected

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_gguf(
        self,
        mock_hf_repo_no_gguf: None,
    ) -> None:
        """Should return empty list for repos with no GGUF files."""
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files("unsloth/adapter-only")
        assert files == []

    @pytest.mark.asyncio
    async def test_sorts_by_size_ascending(
        self,
        mock_hf_repo_multiple_quants: None,
    ) -> None:
        """Should sort results by file size (smallest first)."""
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files("unsloth/multi-quant-model")
        sizes = [f.size_bytes for f in files]
        assert sizes == sorted(sizes)

    @pytest.mark.asyncio
    async def test_raises_for_invalid_repo(
        self,
        mock_hf_repo_not_found: None,
    ) -> None:
        """Should raise RepoNotFoundError for invalid repo_id."""
        fetcher = GGUFMetadataFetcher()
        with pytest.raises(RepoNotFoundError):
            await fetcher.list_gguf_files("invalid/nonexistent")


class TestQuantizationPatterns:
    """Tests for quantization pattern parsing."""

    @pytest.fixture
    def fetcher(self) -> GGUFMetadataFetcher:
        return GGUFMetadataFetcher()

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("model-Q4_K_M.gguf", "Q4_K_M"),
            ("model-Q4_K_S.gguf", "Q4_K_S"),
            ("model-Q5_K_M.gguf", "Q5_K_M"),
            ("model-Q5_K_S.gguf", "Q5_K_S"),
            ("model-Q6_K.gguf", "Q6_K"),
            ("model-Q8_0.gguf", "Q8_0"),
            ("model-Q2_K.gguf", "Q2_K"),
            ("model-Q3_K_M.gguf", "Q3_K_M"),
            ("model-Q3_K_S.gguf", "Q3_K_S"),
            ("model-IQ4_XS.gguf", "IQ4_XS"),
            ("model-IQ4_NL.gguf", "IQ4_NL"),
            # Unsloth dynamic quants
            ("model-Q4_K_XL.gguf", "Q4_K_XL"),
            ("model-Q8_K_XL.gguf", "Q8_K_XL"),
            ("model-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
            ("model-UD-Q2_K_XL.gguf", "UD-Q2_K_XL"),
            # Case insensitive
            ("model-q4_k_m.gguf", "Q4_K_M"),
            ("MODEL-Q4_K_M.GGUF", "Q4_K_M"),
            # No quant in name
            ("model.gguf", None),
            ("some-random-file.gguf", None),
        ],
    )
    def test_parse_quant_patterns(
        self,
        fetcher: GGUFMetadataFetcher,
        filename: str,
        expected: str | None,
    ) -> None:
        """Should correctly parse various quantization patterns."""
        assert fetcher._parse_quant_type(filename) == expected


class TestImatrixFiltering:
    """Tests for filtering out non-model GGUF files."""

    @pytest.mark.asyncio
    async def test_filters_out_imatrix_files(
        self,
        mock_hf_repo_with_imatrix: None,
    ) -> None:
        """Should filter out imatrix.gguf files (calibration data, not models)."""
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files("bartowski/test-model")
        filenames = [f.filename for f in files]
        # imatrix files should be excluded
        assert not any("imatrix" in f.lower() for f in filenames)
        # Actual model files should be included
        assert any("Q4_K_M" in f for f in filenames)

    @pytest.mark.asyncio
    async def test_imatrix_case_insensitive(
        self,
        mock_hf_repo_with_imatrix_variants: None,
    ) -> None:
        """Should filter imatrix files regardless of case."""
        fetcher = GGUFMetadataFetcher()
        files = await fetcher.list_gguf_files("bartowski/test-model")
        filenames = [f.filename for f in files]
        # All imatrix variants should be excluded
        assert not any("imatrix" in f.lower() for f in filenames)


class TestEdgeCases:
    """Edge case tests for GGUF metadata."""

    @pytest.mark.asyncio
    async def test_handles_zero_size_files(self) -> None:
        """Should handle files with zero size gracefully."""
        fetcher = GGUFMetadataFetcher()
        vram = fetcher._estimate_vram(0)
        assert vram == 0

    @pytest.mark.asyncio
    async def test_handles_very_large_files(self) -> None:
        """Should handle very large file sizes (100+ GB)."""
        fetcher = GGUFMetadataFetcher()
        size_bytes = 100 * 1024 * 1024 * 1024  # 100 GB
        vram = fetcher._estimate_vram(size_bytes)
        # Should be ~110 GB (with 1.1 overhead)
        assert vram > 100 * 1024  # > 100 GB in MB
        assert vram < 120 * 1024  # < 120 GB in MB
