// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include <metal_stdlib>
using namespace metal;

namespace m1moemma {

constexpr constant uint kGroup = 64;
constexpr constant uint kStorageN = 256;
constexpr constant uint FN = 4;
constexpr constant uint kSimdgroups = 4;

enum class Pass { GateUp, Down };

template <uint Rows, Pass PassKind>
inline void expert(device const bfloat *grouped_input,
                   device const MoeTileDescriptor *tiles,
                   device const uint *tile_count, device uchar *packed_0,
                   device uchar *packed_1, device uchar *shared_0,
                   device uchar *shared_1, device bfloat *output,
                   constant MoeExpertParams &params, uint2 thread_group, uint simdgroup,
                   uint lane) {
  static_assert(Rows == 8 || Rows == 32, "8- and 32-row expert tiles");
  constexpr uint SGM = Rows == 32 ? 2 : 1;
  constexpr uint SGN = kSimdgroups / SGM;
  constexpr uint FM = Rows / (8 * SGM);
  constexpr bool gateUp = PassKind == Pass::GateUp;
  constexpr uint sgColumns = gateUp ? FN / 2 * 8 : FN * 8;
  constexpr uint tgColumns = SGN * sgColumns;
  static_assert(kStorageN % tgColumns == 0, "left_operand tile never straddles left_operand packed 256-column tile");

  if (thread_group.y >= *tile_count)
    return;
  const MoeTileDescriptor descriptor = tiles[thread_group.y];
  const uint sgn = simdgroup % SGN, sgm = simdgroup / SGN;
  const uint firstRow = sgm * FM * 8;
  if (firstRow >= descriptor.rows)
    return;
  const uint OutputSize = params.output_size, InputSize = params.input_size, groups = InputSize / kGroup;
  const MoeQ4Slab slab_0 =
      moe_q4_slab(packed_0, shared_0, descriptor.expert, params.experts,
                  params.expert_stride_bytes_0, OutputSize, InputSize);
  MoeQ4Slab slab_1 = slab_0;
  if constexpr (gateUp)
    slab_1 = moe_q4_slab(packed_1, shared_1, descriptor.expert, params.experts,
                         params.expert_stride_bytes_1, OutputSize, InputSize);

  const uint qid = lane >> 2;
  const uint fragment_row = (qid & 4) | ((lane >> 1) & 3);
  const uint fragment_column = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint n0 = thread_group.x * tgColumns + sgn * sgColumns;
  const uint tile = n0 / kStorageN;

  device const uchar *laneWeights[FN];
  device const bfloat *laneScales[FN];
  device const bfloat *laneBiases[FN];
#pragma unroll
  for (uint column_fragment = 0; column_fragment < FN; ++column_fragment) {
    const bool up = gateUp && column_fragment >= FN / 2;
    const MoeQ4Slab slab = up ? slab_1 : slab_0;
    const uint output_column = n0 + (gateUp ? column_fragment % (FN / 2) : column_fragment) * 8 + fragment_row;
    const ulong column = ulong(tile) * groups * kStorageN + output_column % kStorageN;
    laneWeights[column_fragment] = slab.weights + column * (kGroup / 2) + (fragment_column / 2) * 8;
    laneScales[column_fragment] = slab.scales + column;
    laneBiases[column_fragment] = slab.biases + column;
  }
  const uint rowBase = firstRow + fragment_column;
  device const bfloat *input =
      grouped_input + (ulong(thread_group.y) * Rows + rowBase) * InputSize + 8 * fragment_row;
  bool live[FM];
#pragma unroll
  for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) live[row_fragment] = firstRow + row_fragment * 8 < descriptor.rows;

  float2 acc[FN][FM];
#pragma unroll
  for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
#pragma unroll
    for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) acc[column_fragment][row_fragment] = float2(0);

  uint2 weight_words[FN];
#pragma unroll
  for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
    weight_words[column_fragment] = *reinterpret_cast<device const uint2 *>(laneWeights[column_fragment]);
  for (uint quant_group = 0; quant_group < groups; ++quant_group) {

    uint4 input_words[FM][2];
    float2 sum[FM];
#pragma unroll
    for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) {
#pragma unroll
      for (uint row_element = 0; row_element < 2; ++row_element) {
        input_words[row_fragment][row_element] = live[row_fragment] ? *reinterpret_cast<device const uint4 *>(
                                input + ulong(row_fragment * 8 + row_element) * InputSize + quant_group * kGroup)
                          : uint4(0);
        float split_index = 0;
#pragma unroll
        for (uint value = 0; value < 4; ++value)
          split_index += as_type<float>(input_words[row_fragment][row_element][value] << 16) + as_type<float>(input_words[row_fragment][row_element][value] & 0xFFFF0000u);
        sum[row_fragment][row_element] = split_index;
      }

      sum[row_fragment] += simd_shuffle_xor(sum[row_fragment], 2);
      sum[row_fragment] += simd_shuffle_xor(sum[row_fragment], 4);
      sum[row_fragment] += simd_shuffle_xor(sum[row_fragment], 16);
    }

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
      const uint shift = (kb & 1) ? 0 : 16;
#pragma unroll
      for (uint row_fragment = 0; row_fragment < FM; ++row_fragment)
        reinterpret_cast<thread float2 &>(right_operand[row_fragment].thread_elements()) =
            float2(as_type<float>((input_words[row_fragment][0][kb >> 1] << shift) & 0xFFFF0000u),
                   as_type<float>((input_words[row_fragment][1][kb >> 1] << shift) & 0xFFFF0000u));
#pragma unroll
      for (uint column_fragment = 0; column_fragment < FN; ++column_fragment) {
        const uint word = kb < 4 ? pairs[column_fragment].x : pairs[column_fragment].y;
        const uint nibbles = (word >> (4 * (kb & 3))) & 0x000F000Fu;

        simdgroup_half8x8 left_operand;
        reinterpret_cast<thread half2 &>(left_operand.thread_elements()) =
            as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h);
#pragma unroll
        for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) {
          if (!live[row_fragment])
            continue;
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
      scale[column_fragment] = float(laneScales[column_fragment][ulong(quant_group) * kStorageN]);
      bias[column_fragment] = float(laneBiases[column_fragment][ulong(quant_group) * kStorageN]);
    }
    if (quant_group + 1 < groups) {
#pragma unroll
      for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
        weight_words[column_fragment] = *reinterpret_cast<device const uint2 *>(
            laneWeights[column_fragment] + ulong(quant_group + 1) * kStorageN * (kGroup / 2));
    }
#pragma unroll
    for (uint row_fragment = 0; row_fragment < FM; ++row_fragment)
#pragma unroll
      for (uint column_fragment = 0; column_fragment < FN; ++column_fragment) {
        acc[column_fragment][row_fragment] = fma(dot[column_fragment][row_fragment], scale[column_fragment], acc[column_fragment][row_fragment]);
        acc[column_fragment][row_fragment] = fma(sum[row_fragment], bias[column_fragment], acc[column_fragment][row_fragment]);
      }
  }

  device bfloat *tileOutput = output + ulong(thread_group.y) * Rows * OutputSize;
#pragma unroll
  for (uint row_fragment = 0; row_fragment < FM; ++row_fragment) {
    if (!live[row_fragment])
      continue;
#pragma unroll
    for (uint row_element = 0; row_element < 2; ++row_element) {
      const uint output_row = rowBase + row_fragment * 8 + row_element;
      if (gateUp) {
#pragma unroll
        for (uint column_fragment = 0; column_fragment < FN / 2; ++column_fragment) {
          const float gate = float(bfloat(acc[column_fragment][row_fragment][row_element]));
          const float up = float(bfloat(acc[column_fragment + FN / 2][row_fragment][row_element]));
          const float value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
          tileOutput[ulong(output_row) * OutputSize + n0 + column_fragment * 8 + fragment_row] = bfloat(value);
        }
      } else {
#pragma unroll
        for (uint column_fragment = 0; column_fragment < FN; ++column_fragment)
          tileOutput[ulong(output_row) * OutputSize + n0 + column_fragment * 8 + fragment_row] = bfloat(acc[column_fragment][row_fragment][row_element]);
      }
    }
  }
}

}

#define MOE_MMA_THREADS                                                        \
  uint2 group [[threadgroup_position_in_grid]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lane [[thread_index_in_simdgroup]]

#define MOE_MMA_KERNELS(Rows)                                                  \
  kernel void moe_expert_gate_up_q4_m1_register_m##Rows(                               \
      device bfloat *grouped_input [[buffer(0)]],                              \
      device const MoeTileDescriptor *tiles [[buffer(1)]],                     \
      device const uint *tile_count [[buffer(2)]],                             \
      device uchar *gate_packed [[buffer(3)]],                                 \
      device uchar *up_packed [[buffer(4)]],                                   \
      device uchar *shared_gate [[buffer(5)]],                                 \
      device uchar *shared_up [[buffer(6)]],                                   \
      device bfloat *output [[buffer(7)]],                                     \
      constant MoeExpertParams &params [[buffer(8)]], MOE_MMA_THREADS) {       \
    m1moemma::expert<Rows, m1moemma::Pass::GateUp>(                                \
        grouped_input, tiles, tile_count, gate_packed, up_packed, shared_gate, \
        shared_up, output, params, group, simd_group, simd_lane);              \
  }                                                                            \
  kernel void moe_expert_down_q4_m1_register_m##Rows(                                  \
      device bfloat *grouped_input [[buffer(0)]],                              \
      device const MoeTileDescriptor *tiles [[buffer(1)]],                     \
      device const uint *tile_count [[buffer(2)]],                             \
      device uchar *down_packed [[buffer(3)]],                                 \
      device uchar *shared_down [[buffer(4)]],                                 \
      device bfloat *output [[buffer(5)]],                                     \
      constant MoeExpertParams &params [[buffer(6)]], MOE_MMA_THREADS) {       \
    m1moemma::expert<Rows, m1moemma::Pass::Down>(                                  \
        grouped_input, tiles, tile_count, down_packed, down_packed,            \
        shared_down, shared_down, output, params, group, simd_group,           \
        simd_lane);                                                            \
  }

MOE_MMA_KERNELS(8)
MOE_MMA_KERNELS(32)
#undef MOE_MMA_KERNELS
#undef MOE_MMA_THREADS
