# Ling 3.0 Flash experiments

| Experiment | Decision / evidence |
| --- | --- |
| Fused MLA prefill kernel (Q-tile + online softmax + PV in one pass, 32×256 exec tile) | Retained (`473c946b`); up to 2.3× prefill vs staged path, suite worst prefill error 9.7e-5 (contract 5e-4). |
| `-UNDEBUG` on `gufo_ling3_flash_kernels` | Retained (`473c946b`): `MlaPrefillFusedKernel` spilled 143 VGPR / 316 B per lane (DRAM scratch) under `-DNDEBUG` only; allocator artifact, asserts absent from the kernel. A/B with `-Rpass-analysis=kernel-resource-usage`; release fatbin now 191 VGPR / 0 spill. |
| KV row staged into registers (`v[KvSlice]`) shared by QK dot and PV update | Retained (`9b1434ca`): harness gain 1.02–1.05× alone; halves shared-memory traffic per (query, key) pair. Not bit-identical to the previous form (compiler contraction moves results ≤ ~4 ulp, maxabs 4e-9 in harness); suite passes. |
| Fused kernel widened to 16 warps, `WarpQ` 4→2, Q-tile held at 32 | Retained (`9b1434ca`): harness gains 1.25–1.43× (best at depth), e2e pp +4.2/+8.4/+13.0/+22.4% at pp2048/4096/8192/16384; release build 118 VGPR, 0 spill, occupancy 12 waves/SIMD. Root lever: 36 KB LDS made 8 warps the resident maximum; the kernel is latency-bound (~2.2 TF/s fp32), not LDS-bound. |
| 16 warps with Q-tile 64 (`WarpQ` 4) | Rejected: 1.27–1.36× — worse than tile-32 at 16K depth (805→846 µs) with equal short-context gains. |
| Shipped-form 8-warp with KV register reuse only | Superseded by the combined 16-warp change (1.02–1.05× alone). |
| WMMA/fp16 attention scores | Deferred: tolerance risk against the 5e-4 prefill contract; not attempted. |

Open items: decode stays ~30 t/s versus the 37 t/s target — the MoE
`RoutedF16GEMMKernel`/GEMV group is the next lever; prefill still decays
802→395 t/s from 2K→16K depth, driven by the same attention kernel
(48% of pp8192 GPU time at `473c946b`). `#pragma unroll 4` on the key loop
now warns (`loop not unrolled`) in the 16-warp form; harmless, revisit if the
loop is touched again.
