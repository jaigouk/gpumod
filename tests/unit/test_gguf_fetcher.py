"""Tests for gpumod.fetchers.gguf — GGUFFetcher."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pytest

from gpumod.fetchers.gguf import GGUFFetcher
from gpumod.models import ModelSource

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers — minimal GGUF file creation
# ---------------------------------------------------------------------------

# GGUF magic bytes: b"GGUF" read as uint32-LE → 0x46554747
_GGUF_MAGIC = 0x46554747

# GGUF metadata value types
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_UINT64 = 6
_GGUF_TYPE_FLOAT32 = 5


def _write_gguf_string(data: bytearray, s: str) -> None:
    """Write a GGUF string (uint64 length + bytes) into the bytearray."""
    encoded = s.encode("utf-8")
    data.extend(struct.pack("<Q", len(encoded)))
    data.extend(encoded)


def _write_gguf_kv_string(data: bytearray, key: str, value: str) -> None:
    """Write a GGUF key-value pair where the value is a string."""
    _write_gguf_string(data, key)
    data.extend(struct.pack("<I", _GGUF_TYPE_STRING))
    _write_gguf_string(data, value)


def _write_gguf_kv_uint32(data: bytearray, key: str, value: int) -> None:
    """Write a GGUF key-value pair where the value is uint32."""
    _write_gguf_string(data, key)
    data.extend(struct.pack("<I", _GGUF_TYPE_UINT32))
    data.extend(struct.pack("<I", value))


def _write_gguf_kv_uint64(data: bytearray, key: str, value: int) -> None:
    """Write a GGUF key-value pair where the value is uint64."""
    _write_gguf_string(data, key)
    data.extend(struct.pack("<I", _GGUF_TYPE_UINT64))
    data.extend(struct.pack("<Q", value))


def _write_gguf_kv_float32(data: bytearray, key: str, value: float) -> None:
    """Write a GGUF key-value pair where the value is float32."""
    _write_gguf_string(data, key)
    data.extend(struct.pack("<I", _GGUF_TYPE_FLOAT32))
    data.extend(struct.pack("<f", value))


def _create_minimal_gguf(
    tmp_path: Path,
    filename: str = "test-model.gguf",
    *,
    version: int = 3,
    architecture: str = "llama",
    file_type: int = 15,  # Q4_K_M
    num_kv: int | None = None,
    extra_size: int = 0,
) -> Path:
    """Create a minimal valid GGUF file for testing.

    The GGUF v3 header format:
    - magic (4 bytes): 0x46475547
    - version (4 bytes): uint32
    - tensor_count (8 bytes): uint64
    - metadata_kv_count (8 bytes): uint64
    - metadata key-value pairs
    """
    data = bytearray()

    # Magic + version
    data.extend(struct.pack("<I", _GGUF_MAGIC))
    data.extend(struct.pack("<I", version))

    # Tensor count (0 for our test files)
    data.extend(struct.pack("<Q", 0))

    # Build metadata KV pairs
    kvs = bytearray()
    kv_count = 0

    # general.architecture
    _write_gguf_kv_string(kvs, "general.architecture", architecture)
    kv_count += 1

    # general.file_type
    _write_gguf_kv_uint32(kvs, "general.file_type", file_type)
    kv_count += 1

    # general.name
    _write_gguf_kv_string(kvs, "general.name", "Test Model")
    kv_count += 1

    actual_kv_count = num_kv if num_kv is not None else kv_count
    data.extend(struct.pack("<Q", actual_kv_count))
    data.extend(kvs)

    # Add extra bytes to simulate a larger file (model weights)
    if extra_size > 0:
        data.extend(b"\x00" * extra_size)

    filepath = tmp_path / filename
    filepath.write_bytes(bytes(data))
    return filepath


def _create_invalid_file(tmp_path: Path, filename: str = "not-a-gguf.bin") -> Path:
    """Create a file that is not a valid GGUF file."""
    filepath = tmp_path / filename
    filepath.write_bytes(b"NOT_GGUF_FILE_HEADER_DATA_HERE")
    return filepath


# ---------------------------------------------------------------------------
# fetch — basic behavior
# ---------------------------------------------------------------------------


class TestFetch:
    """Tests for GGUFFetcher.fetch()."""

    async def test_fetch_returns_model_info_with_gguf_source(self, tmp_path: Path) -> None:
        """fetch() should return ModelInfo with source=GGUF."""
        gguf_path = _create_minimal_gguf(tmp_path)
        fetcher = GGUFFetcher()

        result = await fetcher.fetch(str(gguf_path))

        assert result.source == ModelSource.GGUF

    async def test_fetch_sets_id_from_filename(self, tmp_path: Path) -> None:
        """fetch() should set id from the file name."""
        gguf_path = _create_minimal_gguf(tmp_path, filename="Llama-3-8B-Q4_K_M.gguf")
        fetcher = GGUFFetcher()

        result = await fetcher.fetch(str(gguf_path))

        assert result.id == "Llama-3-8B-Q4_K_M.gguf"

    async def test_fetch_populates_architecture(self, tmp_path: Path) -> None:
        """fetch() should populate architecture from GGUF metadata."""
        gguf_path = _create_minimal_gguf(tmp_path, architecture="mistral")
        fetcher = GGUFFetcher()

        result = await fetcher.fetch(str(gguf_path))

        assert result.architecture == "mistral"

    async def test_fetch_estimates_vram_from_file_size(self, tmp_path: Path) -> None:
        """fetch() should estimate base_vram_mb from file size with overhead."""
        gguf_path = _create_minimal_gguf(tmp_path, extra_size=1024 * 1024 * 100)  # ~100MB
        fetcher = GGUFFetcher()

        result = await fetcher.fetch(str(gguf_path))

        assert result.base_vram_mb is not None
        # File is ~100MB, VRAM estimate should be ~110MB (10% overhead)
        assert result.base_vram_mb > 100

    async def test_fetch_sets_fetched_at(self, tmp_path: Path) -> None:
        """fetch() should set fetched_at to an ISO timestamp."""
        gguf_path = _create_minimal_gguf(tmp_path)
        fetcher = GGUFFetcher()

        result = await fetcher.fetch(str(gguf_path))

        assert result.fetched_at is not None


# ---------------------------------------------------------------------------
# fetch — input validation
# ---------------------------------------------------------------------------


class TestFetchValidation:
    """Tests for GGUFFetcher.fetch() input validation."""

    async def test_fetch_raises_for_missing_file(self) -> None:
        """fetch() should raise FileNotFoundError for nonexistent file."""
        fetcher = GGUFFetcher()
        with pytest.raises(FileNotFoundError):
            await fetcher.fetch("/nonexistent/path/model.gguf")

    async def test_fetch_raises_for_non_gguf_file(self, tmp_path: Path) -> None:
        """fetch() should raise ValueError for files without GGUF magic."""
        bad_file = _create_invalid_file(tmp_path)
        fetcher = GGUFFetcher()

        with pytest.raises(ValueError, match="GGUF"):
            await fetcher.fetch(str(bad_file))

    async def test_fetch_raises_for_directory(self, tmp_path: Path) -> None:
        """fetch() should raise ValueError when given a directory."""
        fetcher = GGUFFetcher()
        with pytest.raises((FileNotFoundError, ValueError)):
            await fetcher.fetch(str(tmp_path))

    async def test_fetch_raises_for_empty_path(self) -> None:
        """fetch() should raise ValueError for empty path."""
        fetcher = GGUFFetcher()
        with pytest.raises((ValueError, FileNotFoundError)):
            await fetcher.fetch("")

    async def test_fetch_raises_for_too_small_file(self, tmp_path: Path) -> None:
        """fetch() should raise ValueError for files smaller than minimum header."""
        tiny_file = tmp_path / "tiny.gguf"
        tiny_file.write_bytes(b"\x47\x55\x46\x46")  # Only magic, no version/metadata
        fetcher = GGUFFetcher()

        with pytest.raises(ValueError, match="too small|GGUF"):
            await fetcher.fetch(str(tiny_file))


# ---------------------------------------------------------------------------
# _parse_header — unit tests
# ---------------------------------------------------------------------------


class TestParseHeader:
    """Tests for GGUFFetcher._parse_header()."""

    def test_parse_header_reads_magic(self, tmp_path: Path) -> None:
        """_parse_header() should read GGUF magic bytes correctly."""
        gguf_path = _create_minimal_gguf(tmp_path)
        fetcher = GGUFFetcher()

        header = fetcher._parse_header(gguf_path)

        assert header["magic"] == _GGUF_MAGIC

    def test_parse_header_reads_version(self, tmp_path: Path) -> None:
        """_parse_header() should read version correctly."""
        gguf_path = _create_minimal_gguf(tmp_path, version=3)
        fetcher = GGUFFetcher()

        header = fetcher._parse_header(gguf_path)

        assert header["version"] == 3

    def test_parse_header_reads_architecture(self, tmp_path: Path) -> None:
        """_parse_header() should read architecture from metadata."""
        gguf_path = _create_minimal_gguf(tmp_path, architecture="llama")
        fetcher = GGUFFetcher()

        header = fetcher._parse_header(gguf_path)

        assert header.get("general.architecture") == "llama"

    def test_parse_header_rejects_invalid_magic(self, tmp_path: Path) -> None:
        """_parse_header() should raise ValueError for non-GGUF files."""
        bad_file = _create_invalid_file(tmp_path)
        fetcher = GGUFFetcher()

        with pytest.raises(ValueError, match="GGUF"):
            fetcher._parse_header(bad_file)

    def test_parse_header_reads_v2_format(self, tmp_path: Path) -> None:
        """_parse_header() should handle GGUF v2 format."""
        gguf_path = _create_minimal_gguf(tmp_path, version=2)
        fetcher = GGUFFetcher()

        header = fetcher._parse_header(gguf_path)

        assert header["version"] == 2


# ---------------------------------------------------------------------------
# _estimate_vram_from_file_size — unit tests
# ---------------------------------------------------------------------------


class TestEstimateVramFromFileSize:
    """Tests for GGUFFetcher._estimate_vram_from_file_size()."""

    def test_estimates_with_overhead(self) -> None:
        """Should add 10% overhead to file size."""
        fetcher = GGUFFetcher()
        # 1GB file = 1024 MB, with 10% overhead = 1126 MB
        result = fetcher._estimate_vram_from_file_size(1024 * 1024 * 1024)
        assert result == 1127  # ceil(1024 * 1.1)

    def test_small_file(self) -> None:
        """Small file should still get overhead applied."""
        fetcher = GGUFFetcher()
        # 100MB file with 10% overhead. Due to floating-point ceil, result is 111.
        result = fetcher._estimate_vram_from_file_size(100 * 1024 * 1024)
        assert result == 111

    def test_zero_size(self) -> None:
        """Zero-size file should return 0."""
        fetcher = GGUFFetcher()
        result = fetcher._estimate_vram_from_file_size(0)
        assert result == 0


# ---------------------------------------------------------------------------
# _read_value — type branch coverage
# ---------------------------------------------------------------------------


class TestReadValue:
    """Tests for GGUFFetcher._read_value() covering all GGUF value types."""

    def test_uint8(self) -> None:
        """Should decode UINT8 (type 0)."""
        data = struct.pack("<B", 42)
        val, new_offset = GGUFFetcher._read_value(data, 0, 0)
        assert val == 42
        assert new_offset == 1

    def test_int8(self) -> None:
        """Should decode INT8 (type 1)."""
        data = struct.pack("<b", -5)
        val, new_offset = GGUFFetcher._read_value(data, 0, 1)
        assert val == -5
        assert new_offset == 1

    def test_uint16(self) -> None:
        """Should decode UINT16 (type 2)."""
        data = struct.pack("<H", 1024)
        val, new_offset = GGUFFetcher._read_value(data, 0, 2)
        assert val == 1024
        assert new_offset == 2

    def test_int16(self) -> None:
        """Should decode INT16 (type 3)."""
        data = struct.pack("<h", -300)
        val, new_offset = GGUFFetcher._read_value(data, 0, 3)
        assert val == -300
        assert new_offset == 2

    def test_uint32(self) -> None:
        """Should decode UINT32 (type 4)."""
        data = struct.pack("<I", 70000)
        val, new_offset = GGUFFetcher._read_value(data, 0, 4)
        assert val == 70000
        assert new_offset == 4

    def test_int32(self) -> None:
        """Should decode INT32 (type 5)."""
        data = struct.pack("<i", -100000)
        val, new_offset = GGUFFetcher._read_value(data, 0, 5)
        assert val == -100000
        assert new_offset == 4

    def test_float32(self) -> None:
        """Should decode FLOAT32 (type 6)."""
        data = struct.pack("<f", 3.14)
        val, new_offset = GGUFFetcher._read_value(data, 0, 6)
        assert abs(val - 3.14) < 0.01
        assert new_offset == 4

    def test_bool_true(self) -> None:
        """Should decode BOOL (type 7) true value."""
        data = struct.pack("<B", 1)
        val, new_offset = GGUFFetcher._read_value(data, 0, 7)
        assert val is True
        assert new_offset == 1

    def test_bool_false(self) -> None:
        """Should decode BOOL (type 7) false value."""
        data = struct.pack("<B", 0)
        val, new_offset = GGUFFetcher._read_value(data, 0, 7)
        assert val is False
        assert new_offset == 1

    def test_uint64(self) -> None:
        """Should decode UINT64 (type 10)."""
        data = struct.pack("<Q", 2**40)
        val, new_offset = GGUFFetcher._read_value(data, 0, 10)
        assert val == 2**40
        assert new_offset == 8

    def test_int64(self) -> None:
        """Should decode INT64 (type 11)."""
        data = struct.pack("<q", -(2**40))
        val, new_offset = GGUFFetcher._read_value(data, 0, 11)
        assert val == -(2**40)
        assert new_offset == 8

    def test_float64(self) -> None:
        """Should decode FLOAT64 (type 12)."""
        data = struct.pack("<d", 2.718281828)
        val, new_offset = GGUFFetcher._read_value(data, 0, 12)
        assert abs(val - 2.718281828) < 1e-6
        assert new_offset == 8

    def test_array_of_uint32(self) -> None:
        """Should decode ARRAY (type 9) of UINT32 values."""
        # Array header: element_type(uint32=4) + count(3)
        data = bytearray()
        data.extend(struct.pack("<I", 4))  # element type = UINT32
        data.extend(struct.pack("<Q", 3))  # 3 elements
        data.extend(struct.pack("<I", 10))
        data.extend(struct.pack("<I", 20))
        data.extend(struct.pack("<I", 30))
        val, new_offset = GGUFFetcher._read_value(bytes(data), 0, 9)
        assert val == [10, 20, 30]
        assert new_offset == len(data)

    def test_unknown_type_returns_none(self) -> None:
        """Should return None for unknown value types."""
        data = b"\x00" * 16
        val, new_offset = GGUFFetcher._read_value(data, 0, 255)
        assert val is None
        assert new_offset == 0


# ---------------------------------------------------------------------------
# _parse_header — edge cases
# ---------------------------------------------------------------------------


class TestParseHeaderEdgeCases:
    """Tests for _parse_header edge cases to cover exception handling."""

    def test_truncated_kv_data_does_not_crash(self, tmp_path: Path) -> None:
        """_parse_header() should handle truncated KV data gracefully."""
        data = bytearray()
        data.extend(struct.pack("<I", _GGUF_MAGIC))
        data.extend(struct.pack("<I", 3))  # version
        data.extend(struct.pack("<Q", 0))  # tensor_count
        data.extend(struct.pack("<Q", 5))  # claim 5 KVs but provide none

        filepath = tmp_path / "truncated.gguf"
        filepath.write_bytes(bytes(data))

        fetcher = GGUFFetcher()
        header = fetcher._parse_header(filepath)
        # Should not raise, just stop parsing
        assert header["magic"] == _GGUF_MAGIC
        assert header["kv_count"] == 5

    def test_offset_beyond_data_stops_iteration(self, tmp_path: Path) -> None:
        """_parse_header() should stop when offset exceeds data length."""
        data = bytearray()
        data.extend(struct.pack("<I", _GGUF_MAGIC))
        data.extend(struct.pack("<I", 3))  # version
        data.extend(struct.pack("<Q", 0))  # tensor_count
        data.extend(struct.pack("<Q", 100))  # claim 100 KVs

        # Add one valid string KV pair, then nothing more
        _write_gguf_kv_string(data, "key1", "value1")

        filepath = tmp_path / "partial.gguf"
        filepath.write_bytes(bytes(data))

        fetcher = GGUFFetcher()
        header = fetcher._parse_header(filepath)
        assert header.get("key1") == "value1"


# ---------------------------------------------------------------------------
# Bug fix: GGUF magic must match real files (b"GGUF" as LE uint32)
# ---------------------------------------------------------------------------


class TestRealGGUFMagic:
    """The GGUF magic bytes are b'GGUF' = [0x47, 0x47, 0x55, 0x46].

    Read as little-endian uint32 this is 0x46554747.
    A previous bug used 0x46475547 which rejected real GGUF files.
    """

    def test_accepts_real_gguf_magic_bytes(self, tmp_path: Path) -> None:
        """_parse_header should accept the real b'GGUF' magic."""
        data = bytearray()
        # Write literal ASCII bytes G G U F
        data.extend(b"GGUF")
        data.extend(struct.pack("<I", 3))  # version
        data.extend(struct.pack("<Q", 0))  # tensor_count
        data.extend(struct.pack("<Q", 0))  # kv_count

        filepath = tmp_path / "real.gguf"
        filepath.write_bytes(bytes(data))

        fetcher = GGUFFetcher()
        header = fetcher._parse_header(filepath)
        assert header["version"] == 3

    def test_rejects_non_gguf_magic(self, tmp_path: Path) -> None:
        """_parse_header should reject files that don't start with b'GGUF'."""
        data = bytearray()
        data.extend(b"GGML")  # wrong magic
        data.extend(struct.pack("<I", 3))
        data.extend(struct.pack("<Q", 0))
        data.extend(struct.pack("<Q", 0))

        filepath = tmp_path / "notgguf.gguf"
        filepath.write_bytes(bytes(data))

        fetcher = GGUFFetcher()
        with pytest.raises(ValueError, match="Invalid GGUF magic"):
            fetcher._parse_header(filepath)


# ---------------------------------------------------------------------------
# Bug fix: quant detection should handle UD- prefixed quants
# ---------------------------------------------------------------------------


class TestQuantDetection:
    """Quant patterns like UD-Q4_K_XL should be detected from filenames."""

    @pytest.mark.asyncio
    async def test_detects_ud_q4_k_xl_quant(self, tmp_path: Path) -> None:
        """Filename containing UD-Q4_K_XL should detect Q4_K_XL quantization."""
        gguf_file = _create_minimal_gguf(
            tmp_path,
            filename="Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf",
        )
        fetcher = GGUFFetcher()
        info = await fetcher.fetch(str(gguf_file))
        assert "Q4_K_XL" in info.quantizations

    @pytest.mark.asyncio
    async def test_detects_ud_q8_k_xl_quant(self, tmp_path: Path) -> None:
        """Filename containing UD-Q8_K_XL should detect Q8_K_XL quantization."""
        gguf_file = _create_minimal_gguf(
            tmp_path,
            filename="Model-UD-Q8_K_XL.gguf",
        )
        fetcher = GGUFFetcher()
        info = await fetcher.fetch(str(gguf_file))
        assert "Q8_K_XL" in info.quantizations
