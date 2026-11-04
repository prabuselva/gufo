// opt-c7u-quant-float4: one wavefront per four Q8_1 blocks, loaded as one
// float4 per lane. The shipped kernel reads a single FP32 per thread; at
// Flint pp8192 that ran the 33-67 MB activation quantizes at ~95 GB/s against
// the ~239 GB/s copy ceiling (STATUS C7u: 588 calls, 263.66 ms). With a
// float4 per lane a wavefront owns 128 elements: block `lane / 8` holds
// element 4 * (lane % 8) + j, so every block is owned by exactly eight
// consecutive lanes and needs only a three-step fmaxf butterfly. The
// per-block max, the int q-sum and the per-element d/id/roundf expressions
// are exact under any evaluation order, so every q byte, scale and sidecar
// value is bit-identical to the shipped kernel (probe
// tools/bench/flint_quant_probe.hip: zero mismatches, -25 % to -55 %).
// Requires k % 128 == 0 and a 16-byte aligned input; callers outside that
// contract keep the shipped kernel.
template<bool StoreActivationSum>
__global__ void QuantizeActivation4ToQ8_1Kernel(const float* __restrict__ x,
                                                void* __restrict__ y,
                                                std::size_t batch,
                                                std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t num_chunks = num_blocks >> 2u;
  const std::size_t wave =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane = threadIdx.x & 31u;
  if (wave >= batch * num_chunks) {
    return;
  }

  const std::size_t tok = wave / num_chunks;
  const std::size_t chunk = wave % num_chunks;
  const std::size_t blk = chunk * 4u + (lane >> 3u);
  const std::size_t sub = lane & 7u;
  const std::size_t tl = tok % kQ8ActTileTokens;
  const std::size_t tt = tok / kQ8ActTileTokens;

  const float4 v =
      reinterpret_cast<const float4*>(x)[(tok * k) / 4u + chunk * 32u + lane];
  const float elems[4] = {v.x, v.y, v.z, v.w};
  float max_abs = fmaxf(fmaxf(fabsf(elems[0]), fabsf(elems[1])),
                        fmaxf(fabsf(elems[2]), fabsf(elems[3])));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;

  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  int qsum = 0;
#pragma unroll
  for (std::size_t j = 0; j < 4u; ++j) {
    const std::size_t pos = sub * 4u + j;
    const std::int8_t q = static_cast<std::int8_t>(roundf(elems[j] * id));
    tile[((pos >> 4u) * 256u) + (tl * 16u) + (pos & 15u)] = q;
    qsum = qsum + static_cast<int>(q);
    if (sub == 0U && j == 0U) {
      *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                (tl * sizeof(float))) = d;
    }
  }
  if constexpr (StoreActivationSum) {
    for (int off = 4; off > 0; off >>= 1) {
      qsum += __shfl_xor(qsum, off);
    }
    if (sub == 0U) {
      float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
      sums[(((tt * num_blocks) + blk) * kQ8ActTileTokens) + tl] =
          d * static_cast<float>(qsum);
    }
  }
}
