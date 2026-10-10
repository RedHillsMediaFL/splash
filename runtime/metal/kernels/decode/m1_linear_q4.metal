// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "metal/kernels/common/q4_sgmatrix.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/abi/Linear.h"

namespace m1q4sgf {
using namespace q4sg;
using namespace sgmatrix;
enum class Epilogue { Affine, Residual, GateUp };

__attribute__((always_inline)) inline void mma(thread float2 &fragment_pair, half2 left_operand, float2 right_operand) {
  simdgroup_half8x8 left_matrix;
  simdgroup_float8x8 right_matrix, accumulator_matrix, output_matrix;
  te(left_matrix) = left_operand;
  te(right_matrix) = right_operand;
  te(accumulator_matrix) = fragment_pair;
  simdgroup_multiply_accumulate(output_matrix, left_matrix, right_matrix, accumulator_matrix);
  fragment_pair = te(output_matrix);
}

template <Epilogue EpilogueKind, uint Lanes, class Out = bfloat>
__attribute__((always_inline)) inline void decode(device const bfloat *table, device const uchar *w0,
                   device const bfloat *sc0, device const bfloat *bi0,
                   device Out *out, device const float *sums,
                   coherent(device) device float *partials, device atomic_uint *counters,
                   device const bfloat *residual, device const uchar *w1,
                   device const bfloat *sc1, device const bfloat *bi1,
                   constant Q4Params &params, uint3 thread_group, uint3 grid, uint thread_index, uint simdgroup, uint lane,
                   threadgroup uint *arrival) {
  constexpr bool gateUp = EpilogueKind == Epilogue::GateUp;
  constexpr uint tileN = gateUp ? 32 : 64;
  const uint OutputSize = params.output_size, InputSize = params.input_size, groups = InputSize / 64;
  const uint splits = grid.y;
  const uint first = thread_group.y * (groups / splits);
  const uint end = thread_group.y + 1 == splits ? groups : first + groups / splits;
  const Lane lane_mapping = lane_map(lane);
  const uint fragment_row = lane_mapping.fm, fragment_column = lane_mapping.fn, fragment_pair = fragment_column / 2;
  const uint base = thread_group.x * tileN + simdgroup * (gateUp ? 8 : 16);
  const uint tile = base / 256;

  auto stream = [&](uint stream_index) { return gateUp ? stream_index : 0u; };
  auto column = [&](uint stream_index) { return base + fragment_row + (gateUp ? 0u : stream_index * 8); };
  device const uchar *tile0 = w0 + ulong(tile) * groups * 8192;
  device const uchar *tile1 = w1 + ulong(tile) * groups * 8192;
  uint2 words[2];
  auto load = [&](uint quant_group) __attribute__((always_inline)) {
#pragma unroll
    for (uint stream_index = 0; stream_index < 2; ++stream_index)
      words[stream_index] = *reinterpret_cast<device const uint2 *>(
          (stream(stream_index) ? tile1 : tile0) + ulong(quant_group) * 8192 + (column(stream_index) % 256) * 32 + fragment_pair * 8);
  };
  float2 acc[2][Lanes];
#pragma unroll
  for (uint stream_index = 0; stream_index < 2; ++stream_index)
#pragma unroll
    for (uint request_lane = 0; request_lane < Lanes; ++request_lane) acc[stream_index][request_lane] = float2(0);
  load(first);
  for (uint quant_group = first; quant_group < end; ++quant_group) {
    uint2 weight_words[2] = {words[0], words[1]};
    if (quant_group + 1 < end) load(quant_group + 1);

    float2 dot[2][Lanes];
#pragma unroll
    for (uint stream_index = 0; stream_index < 2; ++stream_index)
#pragma unroll
      for (uint request_lane = 0; request_lane < Lanes; ++request_lane) dot[stream_index][request_lane] = float2(0);
#pragma unroll
    for (uint h = 0; h < 2; ++h) {

      vec<bfloat, 8> input_words[Lanes];
#pragma unroll
      for (uint request_lane = 0; request_lane < Lanes; ++request_lane)
        input_words[request_lane] = reinterpret_cast<device const vec<bfloat, 8> *>(
            table + ulong(request_lane) * InputSize * kRows + ulong(quant_group) * kXtPerGroup)[(h * 8 + fragment_row) * 4 + fragment_pair];
#pragma unroll
      for (uint split_index = 0; split_index < 4; ++split_index) {
        float2 right_operand[Lanes];
#pragma unroll
        for (uint request_lane = 0; request_lane < Lanes; ++request_lane) right_operand[request_lane] = float2(reinterpret_cast<thread bfloat2 *>(&input_words[request_lane])[split_index]);
#pragma unroll
        for (uint stream_index = 0; stream_index < 2; ++stream_index) {
          const uint nibbles = ((h ? weight_words[stream_index].y : weight_words[stream_index].x) >> (4 * split_index)) & 0x000F000Fu;
          const half2 left_operand = as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h);
#pragma unroll
          for (uint request_lane = 0; request_lane < Lanes; ++request_lane) mma(dot[stream_index][request_lane], left_operand, right_operand[request_lane]);
        }
      }
    }
#pragma unroll
    for (uint stream_index = 0; stream_index < 2; ++stream_index) {
      const ulong prm = (ulong(tile) * groups + quant_group) * 256 + column(stream_index) % 256;
      const float scale = float((stream(stream_index) ? sc1 : sc0)[prm]);
      const float bias = float((stream(stream_index) ? bi1 : bi0)[prm]);
#pragma unroll
      for (uint request_lane = 0; request_lane < Lanes; ++request_lane) {
        device const float *rowSums = sums + ulong(request_lane) * InputSize / 8 + quant_group * kRows + fragment_column;
        acc[stream_index][request_lane] = fma(dot[stream_index][request_lane], scale, acc[stream_index][request_lane]);
        acc[stream_index][request_lane] = fma(float2(rowSums[0], rowSums[1]), bias, acc[stream_index][request_lane]);
      }
    }
  }
  if (splits > 1) {

    auto slot = [&](uint request_lane, uint split_index, uint stream_index) {
      return partials + (ulong(request_lane) * splits * 2 + split_index * 2 + stream(stream_index)) * 8 * OutputSize + column(stream_index);
    };
#pragma unroll
    for (uint stream_index = 0; stream_index < 2; ++stream_index)
#pragma unroll
      for (uint request_lane = 0; request_lane < Lanes; ++request_lane) {
        coherent(device) device float *at = slot(request_lane, thread_group.y, stream_index);
        at[fragment_column * OutputSize] = acc[stream_index][request_lane].x;
        at[(fragment_column + 1) * OutputSize] = acc[stream_index][request_lane].y;
      }

    threadgroup_barrier(mem_flags::mem_device);
    if (thread_index == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope::thread_scope_device);
      *arrival = atomic_fetch_add_explicit(counters + thread_group.x, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope::thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (*arrival != splits - 1) return;
    float2 total[2][Lanes];
#pragma unroll
    for (uint stream_index = 0; stream_index < 2; ++stream_index)
#pragma unroll
      for (uint request_lane = 0; request_lane < Lanes; ++request_lane) total[stream_index][request_lane] = float2(0);
    for (uint split_index = 0; split_index < splits; ++split_index)
#pragma unroll
      for (uint stream_index = 0; stream_index < 2; ++stream_index)
#pragma unroll
        for (uint request_lane = 0; request_lane < Lanes; ++request_lane) {
          coherent(device) device const float *at = slot(request_lane, split_index, stream_index);
          total[stream_index][request_lane] += split_index == thread_group.y ? acc[stream_index][request_lane] : float2(at[fragment_column * OutputSize], at[(fragment_column + 1) * OutputSize]);
        }
#pragma unroll
    for (uint stream_index = 0; stream_index < 2; ++stream_index)
#pragma unroll
      for (uint request_lane = 0; request_lane < Lanes; ++request_lane) acc[stream_index][request_lane] = total[stream_index][request_lane];
    if (thread_index == 0) atomic_store_explicit(counters + thread_group.x, 0u, memory_order_relaxed);
  }
#pragma unroll
  for (uint request_lane = 0; request_lane < Lanes; ++request_lane) {
    device Out *destination = out + ulong(request_lane) * kRows * OutputSize;
    if (gateUp) {
      const uint output_column = column(0);
      const float2 gate = float2(bfloat2(acc[0][request_lane])), up = float2(bfloat2(acc[1][request_lane]));
      const float2 value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
      destination[fragment_column * OutputSize + output_column] = Out(value.x);
      destination[(fragment_column + 1) * OutputSize + output_column] = Out(value.y);
    } else {
      device const bfloat *res = residual + ulong(request_lane) * kRows * OutputSize;
#pragma unroll
      for (uint stream_index = 0; stream_index < 2; ++stream_index) {
        const uint output_column = column(stream_index);
        float2 value = acc[stream_index][request_lane];
        if constexpr (!is_same_v<Out, float>) value = float2(bfloat2(value));
        if (EpilogueKind == Epilogue::Residual)
          value += float2(float(res[fragment_column * OutputSize + output_column]), float(res[(fragment_column + 1) * OutputSize + output_column]));
        destination[fragment_column * OutputSize + output_column] = Out(value.x);
        destination[(fragment_column + 1) * OutputSize + output_column] = Out(value.y);
      }
    }
  }
}
}

#define M1_Q4_INPUTS(Out) device const bfloat *table [[buffer(0)]], device const uchar *weights [[buffer(1)]], device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]], device Out *output [[buffer(4)]], device const float *sums [[buffer(5)]], device coherent(device) float *partials [[buffer(6)]], device atomic_uint *counters [[buffer(7)]]
#define M1_Q4_THREADS uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
#define M1_Q4_KERNELS(Suffix, Lanes) \
kernel void decode_linear_q4_m1_register##Suffix(M1_Q4_INPUTS(bfloat), constant Q4Params &params [[buffer(8)]], M1_Q4_THREADS) { \
  threadgroup uint arrival; \
  m1q4sgf::decode<m1q4sgf::Epilogue::Affine, Lanes>(table, weights, scales, biases, output, sums, partials, counters, table, weights, scales, biases, params, group, grid, thread_index, simdgroup, lane, &arrival); \
} \
kernel void decode_linear_q4_m1_register##Suffix##_f32(M1_Q4_INPUTS(float), constant Q4Params &params [[buffer(8)]], M1_Q4_THREADS) { \
  threadgroup uint arrival; \
  m1q4sgf::decode<m1q4sgf::Epilogue::Affine, Lanes, float>(table, weights, scales, biases, output, sums, partials, counters, table, weights, scales, biases, params, group, grid, thread_index, simdgroup, lane, &arrival); \
} \
kernel void decode_linear_q4_m1_register_residual##Suffix(M1_Q4_INPUTS(bfloat), device const bfloat *residual [[buffer(8)]], constant Q4Params &params [[buffer(9)]], M1_Q4_THREADS) { \
  threadgroup uint arrival; \
  m1q4sgf::decode<m1q4sgf::Epilogue::Residual, Lanes>(table, weights, scales, biases, output, sums, partials, counters, residual, weights, scales, biases, params, group, grid, thread_index, simdgroup, lane, &arrival); \
} \
kernel void decode_linear_q4_m1_register_gate_up##Suffix(M1_Q4_INPUTS(bfloat), device const uchar *up [[buffer(8)]], device const bfloat *up_scales [[buffer(9)]], device const bfloat *up_biases [[buffer(10)]], constant Q4Params &params [[buffer(11)]], M1_Q4_THREADS) { \
  threadgroup uint arrival; \
  m1q4sgf::decode<m1q4sgf::Epilogue::GateUp, Lanes>(table, weights, scales, biases, output, sums, partials, counters, table, up, up_scales, up_biases, params, group, grid, thread_index, simdgroup, lane, &arrival); \
}
M1_Q4_KERNELS(, 1)
M1_Q4_KERNELS(_m16, 2)
M1_Q4_KERNELS(_m24, 3)
M1_Q4_KERNELS(_m32, 4)
#undef M1_Q4_KERNELS
#undef M1_Q4_THREADS
#undef M1_Q4_INPUTS
