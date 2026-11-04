void LaunchZeroQ8ActTailFor(void* q8_1_out, std::size_t batch, std::size_t k,
                            hipStream_t stream) {
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
}

bool TryLaunchQuantPrefillWave32(core::GgmlType type, const void* w,
                                 const void* q8_1_x, float* y,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch < 96 || type != core::GgmlType::kQ8_0) {
    return false;
  }
  constexpr int kBM = 128;
  constexpr int kBN = 128;
  const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                  static_cast<unsigned int>((m + kBM - 1) / kBM));
  WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2, core::GgmlType::kQ8_0, 32>
      <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
  return true;
}

