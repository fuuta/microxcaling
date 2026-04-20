import json
import math
import random
import struct
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from mx.formats import ElemFormat, _get_format_params
from mx.mx_ops import _quantize_mx_debug


MX_ELEM_FORMAT_NAMES = {
    "int8",
    "int4",
    "int2",
    "fp8_e5m2",
    "fp8_e4m3",
    "fp6_e3m2",
    "fp6_e2m3",
    "fp4",
    "fp4_e2m1",
}


def float_to_bf16_bits(v: float) -> int:
    u32 = struct.unpack(">I", struct.pack(">f", float(v)))[0]
    return (u32 >> 16) & 0xFFFF


def float_to_fp16_bits(v: float) -> int:
    return struct.unpack(">H", struct.pack(">e", float(v)))[0]


def float_to_fp32_bits(v: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(v)))[0]


def float_to_bits(v: float, fmt: str) -> int:
    if fmt == "bf16":
        return float_to_bf16_bits(v)
    if fmt == "fp16":
        return float_to_fp16_bits(v)
    if fmt == "fp32":
        return float_to_fp32_bits(v)
    raise ValueError(f"Unsupported storage format: {fmt}")


def _format_params(elem_format):
    ebits, mbits, emax, _, _ = _get_format_params(elem_format)
    if ebits == 0:
        return ebits, mbits, emax, None
    return ebits, mbits, emax, (1 << (ebits - 1)) - 1


def quantized_scaled_to_raw(v: float, ebits: int, mbits: int, bias: Optional[int]) -> int:
    if v == 0.0:
        return 0

    sign = 1 if math.copysign(1.0, float(v)) < 0 else 0
    abs_v = abs(float(v))

    if ebits == 0:
        scale = 1 << (mbits - 2)
        magnitude = int(round(abs_v * scale))
        magnitude = min(magnitude, (1 << (mbits - 1)) - 1)
        return (sign << (mbits - 1)) | magnitude

    if not math.isfinite(abs_v):
        exp_field = (1 << ebits) - 1
        man_field = 1 if math.isnan(abs_v) else 0
        return (sign << (ebits + mbits - 2)) | (exp_field << (mbits - 2)) | man_field

    man_bits = mbits - 2
    exp_unbiased = math.floor(math.log2(abs_v))
    exp_field = exp_unbiased + bias

    if exp_field <= 0:
        # Subnormal: exponent field is zero and the implicit leading 1 is absent.
        subnormal_scale = 2 ** (1 - bias - man_bits)
        man_field = int(round(abs_v / subnormal_scale))
        man_field = min(man_field, (1 << man_bits) - 1)
        exp_field = 0
    else:
        significand = abs_v / (2**exp_unbiased) - 1.0
        man_field = int(round(significand * (1 << man_bits)))
        if man_field == (1 << man_bits):
            man_field = 0
            exp_field += 1
        exp_field = min(exp_field, (1 << ebits) - 1)

    return (sign << (ebits + man_bits)) | (exp_field << man_bits) | man_field


def parse_elem_formats(values: List[str]) -> List[Tuple[str, ElemFormat]]:
    names = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())

    formats = []
    for name in names:
        normalized = name.lower()
        if normalized not in MX_ELEM_FORMAT_NAMES:
            allowed = ", ".join(sorted(MX_ELEM_FORMAT_NAMES))
            raise ValueError(
                f"--elem-formats accepts only MX element formats. "
                f"Got '{name}'. Allowed values: {allowed}"
            )
        elem_format = ElemFormat.from_str(normalized)
        formats.append((normalized, elem_format))
    return formats


def make_random_values(
    rng: random.Random,
    elem_format,
    count: int,
    exponent_min: Optional[int] = None,
    exponent_max: Optional[int] = None,
) -> List[float]:
    ebits, mbits, emax, _ = _format_params(elem_format)

    values = [0.0]
    for _ in range(max(0, count - 1)):
        sign = -1.0 if rng.getrandbits(1) else 1.0

        if ebits == 0:
            magnitude = rng.randint(0, (1 << (mbits - 1)) - 1)
            values.append(sign * magnitude / (1 << (mbits - 2)))
            continue

        emin = 2 - (1 << (ebits - 1))
        lo = emin if exponent_min is None else max(emin, exponent_min)
        hi = emax if exponent_max is None else min(emax, exponent_max)
        if lo > hi:
            raise ValueError(
                f"Random exponent range [{exponent_min}, {exponent_max}] "
                f"does not overlap format range [{emin}, {emax}]"
            )

        exponent = rng.randint(lo, hi)
        mantissa = rng.randint(0, (1 << (mbits - 2)) - 1)
        values.append(sign * (1.0 + mantissa / (1 << (mbits - 2))) * (2**exponent))

    rng.shuffle(values)
    return values


def make_case_from_values(
    name: str,
    values: List[float],
    elem_format_name: str,
    elem_format,
    scale_bits: int = 8,
    block_size: int = 32,
    round_mode: str = "even",
    real_format: str = "bf16",
):
    ebits, mbits, _, bias = _format_params(elem_format)
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
        "real_format": real_format,
        "scale_bits": scale_bits,
        "block_size": block_size,
        "round": round_mode,
        f"input_real": [float_to_bits(v, real_format) for v in x.tolist()],
        "shared_exp": int(shared_exp.tolist()[0]),
        "raw_elements": [quantized_scaled_to_raw(v, ebits, mbits, bias) for v in qs],
        f"dequantized_real": [
            float_to_bits(v, real_format) for v in dequantized.tolist()
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON to this path. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--elem-formats",
        nargs="+",
        default=["fp8_e4m3", "fp8_e5m2"],
        help=(
            "MX element formats to generate. Accepts space- or comma-separated "
            "names, e.g. 'fp6_e3m2 fp6_e2m3 fp8_e4m3'. "
            "Real formats such as bf16/fp16 are selected by --real-format."
        ),
    )
    parser.add_argument(
        "--real-format",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Real value format used for input/dequantized reference fields.",
    )
    parser.add_argument(
        "--storage-format",
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--scale-bits", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--round", default="even", choices=["floor", "nearest", "even"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-cases",
        type=int,
        default=4,
        help="Number of random exponent/mantissa cases to add per element format.",
    )
    parser.add_argument(
        "--random-values-per-case",
        type=int,
        default=32,
        help="Number of values in each random case.",
    )
    parser.add_argument(
        "--random-exponent-min",
        type=int,
        default=None,
        help="Optional lower bound for randomly generated unbiased exponents.",
    )
    parser.add_argument(
        "--random-exponent-max",
        type=int,
        default=None,
        help="Optional upper bound for randomly generated unbiased exponents.",
    )
    parser.add_argument(
        "--no-default-vectors",
        action="store_true",
        help="Only emit random cases; skip the small hand-written vectors.",
    )
    args = parser.parse_args()

    if args.block_size > 0 and args.random_values_per_case > args.block_size:
        parser.error("--random-values-per-case must be <= --block-size for scalar shared_exp output")
    if args.storage_format is not None:
        args.real_format = args.storage_format

    vectors = [
        ("basic_round", [1.0, 1.5, 1.4375]),
        ("signed_mix", [1.0, -1.0, 0.5, -0.5]),
        ("scale_spread", [0.25, 1.0, 4.0, 16.0]),
        ("tiny_mix", [0.0, 0.00390625, 0.0078125, 0.015625]),
    ]

    rng = random.Random(args.seed)
    try:
        elem_formats = parse_elem_formats(args.elem_formats)
    except ValueError as exc:
        parser.error(str(exc))

    cases = []
    for elem_format_name, elem_format in elem_formats:
        if not args.no_default_vectors:
            for suffix, values in vectors:
                cases.append(
                    make_case_from_values(
                        f"{elem_format_name}_{suffix}",
                        values,
                        elem_format_name,
                        elem_format,
                        scale_bits=args.scale_bits,
                        block_size=args.block_size,
                        round_mode=args.round,
                        real_format=args.real_format,
                    )
                )

        for case_idx in range(args.random_cases):
            values = make_random_values(
                rng,
                elem_format,
                args.random_values_per_case,
                exponent_min=args.random_exponent_min,
                exponent_max=args.random_exponent_max,
            )
            cases.append(
                make_case_from_values(
                    f"{elem_format_name}_random_{case_idx}",
                    values,
                    elem_format_name,
                    elem_format,
                    scale_bits=args.scale_bits,
                    block_size=args.block_size,
                    round_mode=args.round,
                    real_format=args.real_format,
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
