#!/usr/bin/env python3
"""Cast every F32 tensor in a safetensors file to F16.

pg_embedder links model weights into the extension binary with include_bytes!, so
file size is binary size. Upstream releases that ship F32 weights double that cost
for no benefit: candle upcasts to F32 at load time either way (DTYPE = F32), so
the F16 file produces the same in-memory model as the F32 one, minus the precision
lost in the cast.

Run this only on weights whose values fit in F16's range. The script refuses to
convert if any value would overflow (|x| > 65504). Values below F16's smallest
normal (6.1e-5) become subnormal or zero; for embedding weights those are noise.

Usage:
    python3 weights/tools/convert_f32_to_f16.py IN.safetensors OUT.safetensors

Stdlib only, no numpy: this has to be runnable wherever the repo is checked out.
"""

import json
import struct
import sys

F16_MAX = 65504.0
CHUNK = 262_144  # elements per pack/unpack call, keeps peak memory bounded


def read_safetensors(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    header_len = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + header_len])
    return header, blob, 8 + header_len


def convert(src, dst):
    header, blob, data_start = read_safetensors(src)
    metadata = header.pop("__metadata__", None)

    # Preserve on-disk tensor order so the output layout mirrors the input.
    tensors = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])

    out_header = {}
    payloads = []
    offset = 0
    converted = skipped = 0

    for name, spec in tensors:
        start, end = spec["data_offsets"]
        raw = blob[data_start + start : data_start + end]

        if spec["dtype"] != "F32":
            # Already narrow (or an integer buffer) — copy through untouched.
            payloads.append(raw)
            out_header[name] = dict(spec, data_offsets=[offset, offset + len(raw)])
            offset += len(raw)
            skipped += 1
            continue

        count = len(raw) // 4
        halves = bytearray()
        for pos in range(0, count, CHUNK):
            take = min(CHUNK, count - pos)
            values = struct.unpack_from("<%df" % take, raw, pos * 4)
            for value in values:
                if abs(value) > F16_MAX:
                    raise SystemExit(
                        "%s: value %r exceeds F16 range; refusing to convert" % (name, value)
                    )
            halves += struct.pack("<%de" % take, *values)

        payloads.append(bytes(halves))
        out_header[name] = {
            "dtype": "F16",
            "shape": spec["shape"],
            "data_offsets": [offset, offset + len(halves)],
        }
        offset += len(halves)
        converted += 1

    if metadata is not None:
        out_header["__metadata__"] = metadata

    header_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    # safetensors wants the data section 8-byte aligned; pad with JSON whitespace.
    header_bytes += b" " * (-len(header_bytes) % 8)

    with open(dst, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for payload in payloads:
            fh.write(payload)

    return converted, skipped


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    converted, skipped = convert(sys.argv[1], sys.argv[2])
    print("converted %d tensors to F16, copied %d unchanged" % (converted, skipped))
