"""GGUF model file info fetcher.

Extracts model metadata from GGUF file headers for VRAM estimation.
Only reads the file header, not the full multi-GB model weights.
"""

from __future__ import annotations

import asyncio
import math
import struct
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gpumod.models import ModelInfo, ModelSource

if TYPE_CHECKING:
    from pathlib import Path

# GGUF magic number: bytes b"GGUF" read as uint32-LE → 0x46554747
_GGUF_MAGIC = 0x46554747

# Minimum header size: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes
_MIN_HEADER_SIZE = 24

# Maximum header read size to prevent memory exhaustion (16 MB)
_MAX_HEADER_READ = 16 * 1024 * 1024

# GGUF metadata value types
_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

# Overhead factor for VRAM estimation from file size
_VRAM_OVERHEAD_FACTOR = 1.1


class GGUFFetcher:
    """Extracts model metadata from GGUF file headers.

    Reads only the binary header of GGUF files to extract architecture,
    quantization type, and other metadata without loading the full model.
    Supports GGUF v2 and v3 formats.
    """

    async def fetch(self, file_path: str) -> ModelInfo:
        """Read GGUF file header to extract model metadata.

        Parameters
        ----------
        file_path:
            Path to the GGUF file.

        Returns
        -------
        ModelInfo
            Model metadata with source=ModelSource.GGUF.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a valid GGUF file or is too small.
        """
        self._validate_path(file_path)

        from pathlib import Path as PathClass

        path = PathClass(file_path)

        header = await asyncio.to_thread(self._parse_header, path)

        # Extract metadata
        architecture = header.get("general.architecture")
        model_name = header.get("general.name")
        file_size = path.stat().st_size

        # Estimate VRAM from file size
        base_vram_mb = self._estimate_vram_from_file_size(file_size)

        # Use filename as ID
        model_id = path.name

        # Extract quantization info from filename if possible
        quantizations: list[str] = []
        name_upper = path.stem.upper()
        for quant in (
            "Q4_K_XL",
            "Q8_K_XL",
            "Q4_K_M",
            "Q4_K_S",
            "Q5_K_M",
            "Q5_K_S",
            "Q6_K",
            "Q8_0",
            "Q4_0",
            "Q4_1",
            "Q5_0",
            "Q5_1",
            "Q2_K",
            "Q3_K_M",
            "Q3_K_S",
            "IQ4_XS",
            "IQ4_NL",
        ):
            if quant in name_upper:
                quantizations.append(quant)
                break

        notes = f"GGUF v{header.get('version', '?')}"
        if model_name:
            notes += f", name={model_name}"

        return ModelInfo(
            id=model_id,
            source=ModelSource.GGUF,
            architecture=architecture,
            base_vram_mb=base_vram_mb,
            quantizations=quantizations,
            fetched_at=datetime.now(tz=UTC).isoformat(),
            notes=notes,
        )

    def _parse_header(self, path: Path) -> dict[str, Any]:
        """Parse GGUF file header (magic + version + metadata).

        Parameters
        ----------
        path:
            Path to the GGUF file.

        Returns
        -------
        dict[str, Any]
            Parsed header with magic, version, and metadata key-value pairs.

        Raises
        ------
        ValueError
            If the file is not a valid GGUF file.
        """
        file_size = path.stat().st_size
        if file_size < _MIN_HEADER_SIZE:
            msg = f"File too small to be a valid GGUF file ({file_size} bytes)"
            raise ValueError(msg)

        read_size = min(file_size, _MAX_HEADER_READ)

        with path.open("rb") as f:
            data = f.read(read_size)

        offset = 0
        result: dict[str, Any] = {}

        # Read magic (4 bytes, uint32 LE)
        magic = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        if magic != _GGUF_MAGIC:
            msg = f"Invalid GGUF magic: 0x{magic:08X} (expected 0x{_GGUF_MAGIC:08X})"
            raise ValueError(msg)

        result["magic"] = magic

        # Read version (4 bytes, uint32 LE)
        version = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        result["version"] = version

        # Read tensor_count (8 bytes, uint64 LE)
        tensor_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        result["tensor_count"] = tensor_count

        # Read metadata_kv_count (8 bytes, uint64 LE)
        kv_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        result["kv_count"] = kv_count

        # Parse metadata key-value pairs
        for _ in range(kv_count):
            if offset >= len(data):
                break
            try:
                key, offset = self._read_string(data, offset)
                value_type = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                value, offset = self._read_value(data, offset, value_type)
                result[key] = value
            except (struct.error, IndexError):
                # Ran out of data in the header buffer
                break

        return result

    def _estimate_vram_from_file_size(self, file_size_bytes: int) -> int:
        """Estimate VRAM from GGUF file size.

        GGUF file size approximates the model weights in quantized form.
        Adds 10% overhead for runtime buffers, KV cache scratch, etc.

        Parameters
        ----------
        file_size_bytes:
            Size of the GGUF file in bytes.

        Returns
        -------
        int
            Estimated VRAM in megabytes.
        """
        if file_size_bytes == 0:
            return 0
        size_mb = file_size_bytes / (1024 * 1024)
        return math.ceil(size_mb * _VRAM_OVERHEAD_FACTOR)

    @staticmethod
    def _validate_path(file_path: str) -> None:
        """Validate the file path.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the path is empty or points to a directory.
        """
        if not file_path:
            msg = "file_path must not be empty"
            raise ValueError(msg)

        from pathlib import Path as PathClass

        path = PathClass(file_path)

        if not path.exists():
            msg = f"GGUF file not found: {file_path!r}"
            raise FileNotFoundError(msg)

        if not path.is_file():
            msg = f"Path is not a regular file: {file_path!r}"
            raise ValueError(msg)

    @staticmethod
    def _read_string(data: bytes, offset: int) -> tuple[str, int]:
        """Read a GGUF string (uint64 length + UTF-8 bytes).

        Returns
        -------
        tuple[str, int]
            The decoded string and the new offset.
        """
        length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        value = data[offset : offset + length].decode("utf-8", errors="replace")
        offset += length
        return value, offset

    @staticmethod
    def _read_value(data: bytes, offset: int, value_type: int) -> tuple[Any, int]:
        """Read a GGUF metadata value based on its type.

        Returns
        -------
        tuple[Any, int]
            The decoded value and the new offset.
        """
        if value_type == _TYPE_UINT8:
            val = struct.unpack_from("<B", data, offset)[0]
            return val, offset + 1
        if value_type == _TYPE_INT8:
            val = struct.unpack_from("<b", data, offset)[0]
            return val, offset + 1
        if value_type == _TYPE_UINT16:
            val = struct.unpack_from("<H", data, offset)[0]
            return val, offset + 2
        if value_type == _TYPE_INT16:
            val = struct.unpack_from("<h", data, offset)[0]
            return val, offset + 2
        if value_type == _TYPE_UINT32:
            val = struct.unpack_from("<I", data, offset)[0]
            return val, offset + 4
        if value_type == _TYPE_INT32:
            val = struct.unpack_from("<i", data, offset)[0]
            return val, offset + 4
        if value_type == _TYPE_FLOAT32:
            val = struct.unpack_from("<f", data, offset)[0]
            return val, offset + 4
        if value_type == _TYPE_BOOL:
            val = struct.unpack_from("<B", data, offset)[0] != 0
            return val, offset + 1
        if value_type == _TYPE_STRING:
            return GGUFFetcher._read_string(data, offset)
        if value_type == _TYPE_UINT64:
            val = struct.unpack_from("<Q", data, offset)[0]
            return val, offset + 8
        if value_type == _TYPE_INT64:
            val = struct.unpack_from("<q", data, offset)[0]
            return val, offset + 8
        if value_type == _TYPE_FLOAT64:
            val = struct.unpack_from("<d", data, offset)[0]
            return val, offset + 8
        if value_type == _TYPE_ARRAY:
            # Read array type and count
            arr_type = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            arr_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            arr: list[Any] = []
            for _ in range(arr_len):
                elem, offset = GGUFFetcher._read_value(data, offset, arr_type)
                arr.append(elem)
            return arr, offset

        # Unknown type: skip
        return None, offset
