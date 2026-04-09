import json
import struct
import argparse
from pathlib import Path

import torch

from mx.formats import ElemFormat
from mx.mx_ops import _quantize_mx_debug


def float_to_bf16_bits(v: float) -> int:
    u32 = struct.unpack(">I", struct.pack(">f", float(v)))[0]
    return (u32 >> 16) & 0xFFFF


def quantized_scaled_to_raw(v: float, ebits: int, mbits: int, bias: int) -> int:
    if v == 0.0:
        return 0
    u32 = struct.unpack(">I", struct.pack(">f", float(v)))[0]
    sign = (u32 >> 31) & 0x1
    exp = (u32 >> 23) & 0xFF
    frac = u32 & 0x7FFFFF
    unbiased = exp - 127
    dst_exp = unbiased + bias
    dst_man = (frac >> (23 - mbits)) & ((1 << mbits) - 1)
    return (sign << (ebits + mbits)) | (dst_exp << mbits) | dst_man


def make_case_from_values(
    name: str,
    values: list,
    elem_format_name: str,
    elem_format,
    ebits: int,
    mbits: int,
    bias: int,
    scale_bits: int = 8,
    block_size: int = 32,
    round_mode: str = "even",
):
    x = torch.tensor(values, dtype=torch.float32)
    shared_exp, _, quantized_scaled, dequantized = _quantize_mx_debug(
        x,
        scale_bits=scale_bits,
        elem_format=elem_format,
        shared_exp_method="max",
        axes=[0],
        block_size=block_size,
        round=round_mode,
        flush_fp32_subnorms=False,
        custom_cuda=False,
    )
    qs = quantized_scaled.tolist()
    return {
        "name": name,
        "elem_format": elem_format_name,
        "scale_bits": scale_bits,
        "block_size": block_size,
        "round": round_mode,
        "input_bf16": [float_to_bf16_bits(v) for v in x.tolist()],
        "shared_exp": int(shared_exp.tolist()[0]),
        "raw_elements": [quantized_scaled_to_raw(v, ebits, mbits, bias) for v in qs],
        "dequantized_bf16": [float_to_bf16_bits(v) for v in dequantized.tolist()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON to this path. If omitted, print to stdout.",
    )
    args = parser.parse_args()

    vectors = [
        ("basic_round", [1.0, 1.5, 1.4375]),
        ("signed_mix", [1.0, -1.0, 0.5, -0.5]),
        ("scale_spread", [0.25, 1.0, 4.0, 16.0]),
        ("tiny_mix", [0.0, 0.00390625, 0.0078125, 0.015625]),
    ]

    cases = []
    for suffix, values in vectors:
        cases.append(
            make_case_from_values(
                f"fp8_e4m3_{suffix}",
                values,
                "fp8_e4m3",
                ElemFormat.fp8_e4m3,
                4,
                3,
                7,
            )
        )
        cases.append(
            make_case_from_values(
                f"fp8_e5m2_{suffix}",
                values,
                "fp8_e5m2",
                ElemFormat.fp8_e5m2,
                5,
                2,
                15,
            )
        )

    out = {"cases": cases}
    text = json.dumps(out, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
