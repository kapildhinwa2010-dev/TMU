from __future__ import annotations

from typing import Any, Callable, Tuple

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void record_clock64_kernel(unsigned long long* out) {
    *out = clock64();
}

__global__ void record_globaltimer_kernel(unsigned long long* out) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *out = t;
}

void record_clock64(torch::Tensor out) {
    TORCH_CHECK(out.is_cuda(), "Tensor must be CUDA");
    TORCH_CHECK(out.numel() >= 1, "Tensor must have at least 1 element");
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto ptr = reinterpret_cast<unsigned long long*>(out.data_ptr());
    record_clock64_kernel<<<1, 1, 0, stream>>>(ptr);
}

void record_globaltimer(torch::Tensor out) {
    TORCH_CHECK(out.is_cuda(), "Tensor must be CUDA");
    TORCH_CHECK(out.numel() >= 1, "Tensor must have at least 1 element");
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto ptr = reinterpret_cast<unsigned long long*>(out.data_ptr());
    record_globaltimer_kernel<<<1, 1, 0, stream>>>(ptr);
}
"""

_CPP_SRC = r"""
void record_clock64(torch::Tensor out);
void record_globaltimer(torch::Tensor out);
"""


class CudaCycleTimer:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA available nahi hai.")

        from torch.utils.cpp_extension import load_inline

        self.ext = load_inline(
            name="tmu_cuda_cycle_timer_ext",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["record_clock64", "record_globaltimer"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )

    def measure(self, fn: Callable[[], Any]) -> Tuple[Any, dict]:
        device = torch.device("cuda")

        start_clock = torch.zeros(1, dtype=torch.int64, device=device)
        end_clock = torch.zeros(1, dtype=torch.int64, device=device)

        start_gt = torch.zeros(1, dtype=torch.int64, device=device)
        end_gt = torch.zeros(1, dtype=torch.int64, device=device)

        self.ext.record_clock64(start_clock)
        self.ext.record_globaltimer(start_gt)

        out = fn()

        self.ext.record_clock64(end_clock)
        self.ext.record_globaltimer(end_gt)

        torch.cuda.synchronize()

        cycles = int(end_clock.item() - start_clock.item())
        globaltimer_ns = int(end_gt.item() - start_gt.item())

        info = {
            "cycles": cycles,
            "globaltimer_ns": globaltimer_ns,
            "globaltimer_us": globaltimer_ns / 1000.0,
            "globaltimer_ms": globaltimer_ns / 1_000_000.0,
        }

        return out, info
