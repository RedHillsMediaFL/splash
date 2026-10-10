// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "metal/abi/Linear.h"
#include <metal_stdlib>
using namespace metal;

namespace m1q4mm {

constexpr constant uint kRows = 32;
constexpr constant uint kGroup = 64;
constexpr constant uint kStorageN = 256;
constexpr constant uint FN = 4, FM = 2;
constexpr constant uint SGN = 2, SGM = 2;
constexpr constant uint kColumns = SGN * FN * 8;
constexpr constant uint kSimdgroups = SGN * SGM;
static_assert(SGM * FM * 8 == kRows && kStorageN % kColumns == 0 && kColumns % kGroup == 0);

enum class Epilogue { Affine, Residual, UpWithGate };

template <Epilogue EpilogueKind>
inline void project(device const bfloat *input, device const uchar *weights,
                    device const bfloat *scales, device const bfloat *biases,
                    device bfloat *output, device const bfloat *auxiliary,
                    device const float *sums, uint OutputSize, uint InputSize, uint planeInputSize, uint2 thread_group,
                    uint simdgroup, uint lane) {
  const uint groups = InputSize / kGroup;
  const uint planeGroups = planeInputSize / kGroup;
  const uint n0 = thread_group.y * kColumns;
  const uint tile = n0 / kStorageN;
  const uint sgn = simdgroup % SGN, sgm = simdgroup / SGN;

  const uint qid = lane >> 2;
  const uint fragment_row = (qid & 4) | ((lane >> 1) & 3);
  const uint fragment_column = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint nBase = sgn * FN * 8;
  const uint rowBase = sgm * FM * 8 + fragment_column;
  const uint nLocal = (n0 + nBase) % kStorageN + fragment_row;
  device const uchar *laneWeights = weights +
      (ulong(tile) * planeGroups * kStorageN + nLocal) * (kGroup / 2) + (fragment_column / 2) * 8;
  device const bfloat *laneScales = scales + ulong(tile) * planeGroups * kStorageN + nLocal;
  device const bfloat *laneBiases = biases + ulong(tile) * planeGroups * kStorageN + nLocal;
  input += (ulong(thread_group.x) * kRows + rowBase) * InputSize + 8 * fragment_row;
  sums += ulong(thread_group.x) * kRows * groups + rowBase;
  output += ulong(thread_group.x) * kRows * OutputSize;
  auxiliary += ulong(thread_group.x) * kRows * OutputSize;

  float2 acc[FN][FM];
#pragma unroll
  for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
#pragma unroll
    for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) acc[column_fragment][row_fragment] = float2(0);

  uint2 weight_words[FN];
#pragma unroll
  for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
    weight_words[column_fragment] = *reinterpret_cast<device const uint2 *>(laneWeights + column_fragment * 8 * 32);
  for (uint quant_group = 0; quant_group < groups; ++quant_group) {

    uint4 input_words[FM][2];
#pragma unroll
    for (uint row_fragment = 0; row_fragment < FM; ++row_fragment)
#pragma unroll
      for (uint row_element = 0; row_element < 2; ++row_element)
        input_words[row_fragment][row_element] = *reinterpret_cast<device const uint4 *>(
            input + ulong(row_fragment * 8 + row_element) * InputSize + quant_group * kGroup);

    uint2 pairs[FN];
#pragma unroll
    for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
      pairs[column_fragment] = uint2((weight_words[column_fragment].x & 0xFFFFu) | (weight_words[column_fragment].y << 16),
                       (weight_words[column_fragment].x >> 16) | (weight_words[column_fragment].y & 0xFFFF0000u));

    float2 dot[FN][FM];
#pragma unroll
    for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
#pragma unroll
      for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) dot[column_fragment][row_fragment] = float2(0);
#pragma unroll
    for (uint kb = 0; kb < 8; ++kb) {
      simdgroup_float8x8 right_operand[FM];
#pragma unroll
      for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) {
        const uint shift = (kb & 1) ? 0 : 16;
        reinterpret_cast<thread float2 &>(right_operand[row_fragment].thread_elements()) =
            float2(as_type<float>((input_words[row_fragment][0][kb >> 1] << shift) & 0xFFFF0000u),
                   as_type<float>((input_words[row_fragment][1][kb >> 1] << shift) & 0xFFFF0000u));
      }
#pragma unroll
      for (uint column_fragment = 0; column_fragment < FN; ++column_fragment) {
        const uint word = kb < 4 ? pairs[column_fragment].x : pairs[column_fragment].y;
        const uint nibbles = (word >> (4 * (kb & 3))) & 0x000F000Fu;

        simdgroup_half8x8 left_operand;
        reinterpret_cast<thread half2 &>(left_operand.thread_elements()) =
            as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h);
#pragma unroll
        for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) {
          simdgroup_float8x8 fragment_pair, d;
          reinterpret_cast<thread float2 &>(fragment_pair.thread_elements()) = dot[column_fragment][row_fragment];
          simdgroup_multiply_accumulate(d, left_operand, right_operand[row_fragment], fragment_pair);
          dot[column_fragment][row_fragment] = reinterpret_cast<thread float2 &>(d.thread_elements());
        }
      }
    }
    float scale[FN], bias[FN];
#pragma unroll
    for (uint column_fragment = 0; column_fragment < FN; ++column_fragment) {
      scale[column_fragment] = float(laneScales[ulong(quant_group) * kStorageN + column_fragment * 8]);
      bias[column_fragment] = float(laneBiases[ulong(quant_group) * kStorageN + column_fragment * 8]);
    }
    if (quant_group + 1 < groups) {
#pragma unroll
      for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
        weight_words[column_fragment] = *reinterpret_cast<device const uint2 *>(
            laneWeights + (ulong(quant_group + 1) * kStorageN + column_fragment * 8) * 32);
    }
#pragma unroll
    for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) {
      const float2 sum = float2(sums[quant_group * kRows + row_fragment * 8], sums[quant_group * kRows + row_fragment * 8 + 1]);
#pragma unroll
      for (uint column_fragment = 0; column_fragment < FN; ++column_fragment) {
        acc[column_fragment][row_fragment] = fma(dot[column_fragment][row_fragment], scale[column_fragment], acc[column_fragment][row_fragment]);
        acc[column_fragment][row_fragment] = fma(sum, bias[column_fragment], acc[column_fragment][row_fragment]);
      }
    }
  }

#pragma unroll
  for (uint column_fragment = 0; column_fragment < FN; ++column_fragment) {
    const uint output_column = n0 + nBase + column_fragment * 8 + fragment_row;
#pragma unroll
    for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) {
#pragma unroll
      for (uint row_element = 0; row_element < 2; ++row_element) {
        const uint output_row = rowBase + row_fragment * 8 + row_element;
        float value = float(bfloat(acc[column_fragment][row_fragment][row_element]));
        if (EpilogueKind == Epilogue::UpWithGate) {
          const float gate = float(auxiliary[output_row * OutputSize + output_column]);
          value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
        } else if (EpilogueKind == Epilogue::Residual) {
          value += float(auxiliary[output_row * OutputSize + output_column]);
        }
        output[output_row * OutputSize + output_column] = bfloat(value);
      }
    }
  }
}

inline void write_output_sums(device const bfloat *output, device float *outputSums,
                              uint OutputSize, uint2 thread_group, uint simdgroup, uint lane) {
  constexpr uint Groups = kColumns / kGroup;
  threadgroup_barrier(mem_flags::mem_device);
  output += ulong(thread_group.x) * kRows * OutputSize;
  outputSums += ulong(thread_group.x) * kRows * (OutputSize / kGroup);
  const uint n0 = thread_group.y * kColumns;
  for (uint task = simdgroup; task < kRows * Groups; task += kSimdgroups) {
    const uint row = task / Groups, local = task % Groups;
    const uint origin = row * OutputSize + n0 + local * kGroup + lane;
    const float sum = simd_sum(float(output[origin]) + float(output[origin + 32]));
    if (lane == 0) outputSums[(n0 / kGroup + local) * kRows + row] = sum;
  }
}

}

#define M1_LINEAR_THREADS uint2 group [[threadgroup_position_in_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
kernel void prefill_linear_q4_m1_register(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    constant Q4Params &params [[buffer(6)]], M1_LINEAR_THREADS) {
  m1q4mm::project<m1q4mm::Epilogue::Affine>(input, weights, scales, biases, output, output,
      sums, params.output_size, params.input_size, params.input_size, group, simdgroup, lane);
}
kernel void prefill_linear_q4_m1_register_residual(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device const bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], constant Q4Params &params [[buffer(7)]], M1_LINEAR_THREADS) {
  m1q4mm::project<m1q4mm::Epilogue::Residual>(input, weights, scales, biases, output, residual,
      sums, params.output_size, params.input_size, params.input_size, group, simdgroup, lane);
}
kernel void prefill_linear_q4_m1_register_residual_leading_inputs(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device const bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], constant Q4PrefillLeadingParams &params [[buffer(7)]], M1_LINEAR_THREADS) {
  m1q4mm::project<m1q4mm::Epilogue::Residual>(input, weights, scales, biases, output, residual,
      sums, params.matrix.output_size, params.matrix.input_size, params.plane_input_size, group, simdgroup, lane);
}
kernel void prefill_linear_q4_m1_register_up_silu_sums(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]],
    device const bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], device float *output_sums [[buffer(7)]],
    constant Q4Params &params [[buffer(8)]], M1_LINEAR_THREADS) {
  m1q4mm::project<m1q4mm::Epilogue::UpWithGate>(input, weights, scales, biases, output, gate,
      sums, params.output_size, params.input_size, params.input_size, group, simdgroup, lane);
  m1q4mm::write_output_sums(output, output_sums, params.output_size, group, simdgroup, lane);
}
#undef M1_LINEAR_THREADS
