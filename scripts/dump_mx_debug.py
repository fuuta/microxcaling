import torch

from mx.formats import ElemFormat
from mx.mx_ops import _quantize_mx_debug


def run_case(x, elem_format):
    shared_exp, scaled_input, quantized_scaled, dequantized = _quantize_mx_debug(
        x,
        scale_bits=8,
        elem_format=elem_format,
        shared_exp_method="max",
        axes=[0],
        block_size=32,
        round="even",
        flush_fp32_subnorms=False,
        custom_cuda=False,
    )
    print("x               =", x.tolist())
    print("shared_exp      =", shared_exp.tolist())
    print("scaled_input    =", scaled_input.tolist())
    print("quantized_scaled=", quantized_scaled.tolist())
    print("dequantized     =", dequantized.tolist())
    print()


run_case(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float32), ElemFormat.fp8_e4m3)
run_case(torch.tensor([1.0, 1.5, 1.4375], dtype=torch.float32), ElemFormat.fp8_e4m3)
run_case(torch.tensor([1.0, 1.5, 1.4375], dtype=torch.float32), ElemFormat.fp8_e5m2)
