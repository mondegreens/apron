"""RFC 8785 JSON Canonicalization Scheme and multihash-prefixed SHA-256 digest.

INV-43: every record, manifest and evidence release serializes to canonical
form before hashing. Published test vectors reproduce every digest from an
independent implementation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

MULTIHASH_SHA2_256: bytes = b"\x12\x20"


def _serialize_float(value: float) -> str:
    """Serialize a float per RFC 8785 §3.2.2.3 (ES6 Number.prototype.toString).

    ES6 finds the shortest decimal s with k digits and exponent n such that
    s * 10^(n-k) equals the IEEE 754 value, then picks notation by n:
      k <= n <= 21        →  integer form (digits + trailing zeros)
      0 < n < k           →  decimal with dot inside the digits
      -6 < n <= 0         →  0.000...digits
      otherwise           →  exponential
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"RFC 8785 does not permit NaN or Infinity: {value}")

    if value == 0.0:
        return "0"

    r = repr(value)
    sign = "-" if r.startswith("-") else ""
    abs_r = r.lstrip("-")

    if "e" in abs_r or "E" in abs_r:
        parts = abs_r.lower().split("e")
        mant = parts[0]
        exp = int(parts[1])
    else:
        mant = abs_r
        exp = 0

    if "." in mant:
        int_part, frac_part = mant.split(".")
        frac_part = frac_part.rstrip("0")
    else:
        int_part, frac_part = mant, ""

    digits = int_part.lstrip("0") + frac_part
    if not digits:
        return "0"
    k = len(digits)
    n = exp + len(int_part.lstrip("0") or "0")
    if int_part == "0":
        n = exp - (len(mant.split(".")[1]) - len(mant.split(".")[1].lstrip("0")))
        digits = mant.split(".")[1].lstrip("0").rstrip("0")
        k = len(digits)

    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n < k:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    m = digits if k == 1 else digits[0] + "." + digits[1:].rstrip("0")
    e = n - 1
    return sign + m + ("e+" if e > 0 else "e") + str(e)


def _serialize_string(value: str) -> str:
    """Serialize a string per RFC 8785 §3.2.2.2."""
    result = ['"']
    for ch in value:
        cp = ord(ch)
        if ch == "\\":
            result.append("\\\\")
        elif ch == '"':
            result.append('\\"')
        elif ch == "\b":
            result.append("\\b")
        elif ch == "\f":
            result.append("\\f")
        elif ch == "\n":
            result.append("\\n")
        elif ch == "\r":
            result.append("\\r")
        elif ch == "\t":
            result.append("\\t")
        elif cp < 0x20:
            result.append(f"\\u{cp:04x}")
        else:
            result.append(ch)
    result.append('"')
    return "".join(result)


def _serialize_value(value: Any) -> str:
    """Serialize a JSON value per RFC 8785."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _serialize_float(value)
    if isinstance(value, str):
        return _serialize_string(value)
    if isinstance(value, list):
        elements = ",".join(_serialize_value(item) for item in value)
        return f"[{elements}]"
    if isinstance(value, dict):
        # RFC 8785 §3.2.3: sort keys by UTF-16 code unit order
        sorted_keys = sorted(value.keys(), key=lambda k: k.encode("utf-16-le"))
        members = ",".join(
            f"{_serialize_string(k)}:{_serialize_value(value[k])}" for k in sorted_keys
        )
        return f"{{{members}}}"
    raise TypeError(f"Cannot canonicalize type {type(value).__name__}")


def canonicalize(obj: Any) -> bytes:
    """Serialize a JSON-compatible value to RFC 8785 canonical form (UTF-8 bytes)."""
    return _serialize_value(obj).encode("utf-8")


def digest(data: bytes) -> bytes:
    """SHA-256 digest with multihash prefix (0x12 = sha2-256, 0x20 = 32 bytes)."""
    raw = hashlib.sha256(data).digest()
    return MULTIHASH_SHA2_256 + raw


def digest_hex(data: bytes) -> str:
    """SHA-256 digest with multihash prefix, as a hex string."""
    return digest(data).hex()


def record_digest(obj: Any) -> bytes:
    """Canonicalize a record and return its multihash-prefixed SHA-256 digest."""
    return digest(canonicalize(obj))


def record_digest_hex(obj: Any) -> str:
    """Canonicalize a record and return its digest as a hex string."""
    return digest_hex(canonicalize(obj))
