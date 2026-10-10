// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#pragma once

#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/sgmatrix.h"

namespace m1sgf {
using sgmatrix::Lane;
using sgmatrix::lane_map;
using sgmatrix::te;

__attribute__((always_inline)) inline void mma(thread float2 &accumulator, half2 left, float2 right) {
  simdgroup_half8x8 left_matrix;
  simdgroup_float8x8 right_matrix, accumulator_matrix, output_matrix;
  te(left_matrix) = left;
  te(right_matrix) = right;
  te(accumulator_matrix) = accumulator;
  simdgroup_multiply_accumulate(output_matrix, left_matrix, right_matrix, accumulator_matrix);
  accumulator = te(output_matrix);
}

__attribute__((always_inline)) inline half2 int8_pair(uint w) {
  return as_type<half2>(((w & 0x00FF00FFu) ^ 0x00800080u) | 0x64006400u) -
         half2(1152.0h);
}

__attribute__((always_inline)) inline float2 bfloat_pair(uint p) {
  return float2(as_type<float>(p << 16), as_type<float>(p & 0xFFFF0000u));
}

inline uint page_token(uint k, uint score_step) {
  return 8 * (k >> 1) + 4 * (score_step >> 1) + (score_step & 1) + 2 * (k & 1);
}

inline uint block_dimension(uint k, uint product_step) {
  return 16 * (k >> 1) + 4 * (product_step >> 1) + (product_step & 1) + 2 * (k & 1);
}

template <uint KVHeads, uint QueryHeadsPerKVHead>
__attribute__((always_inline)) inline void tile(
    device const bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    uint sg, uint lane) {
  constexpr uint QueryGroupHeads = QueryHeadsPerKVHead;
  constexpr uint Rows = 8;
  constexpr uint FusedRows = Rows * QueryGroupHeads;
  constexpr uint PageTokens = SplashKvPageTokens;
  constexpr uint HeadDimension = SplashKvHeadDimension;
  static_assert(PageTokens == 32 && HeadDimension == 256, "fragment maps assume Page32 and HeadDimension = 256");
  static_assert(SPLASH_TARGET_VERIFY_ROWS == Rows &&
                    SPLASH_PREFILL_ATTENTION_TILE_ROWS == Rows,
                "one simdgroup per QueryGroupHeads fused rows of an eight-row tile");
  const uint visible = committed_tokens + active_rows;
  const uint pages = splash_attention_pages(visible);
  const uint per_split = splash_attention_pages_per_split(pages, splits);
  const uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  const uint page_end = min(pages, page_begin + per_split);
  const Lane lane_mapping = lane_map(lane);
  const uint fragment_row = lane_mapping.fm, fragment_column = lane_mapping.fn;

  const uint row = sg * 8 + fragment_column;
  const uint2 limit(
      min(visible, committed_tokens + min(row / QueryGroupHeads, active_rows - 1) + 1),
      min(visible, committed_tokens + min((row + 1) / QueryGroupHeads, active_rows - 1) + 1));

  uint queries[4][8];
  {
    device const ushort *q0 =
        reinterpret_cast<device const ushort *>(tile_queries) + row * HeadDimension;
    device const ushort *q1 = q0 + HeadDimension;
#pragma unroll
    for (uint dimension_block = 0; dimension_block < 4; ++dimension_block)
#pragma unroll
      for (uint product_step = 0; product_step < 8; ++product_step) {
        const uint dimension_tile = 64 * dimension_block + block_dimension(fragment_row, product_step);
        queries[dimension_block][product_step] = uint(q0[dimension_tile]) | (uint(q1[dimension_tile]) << 16);
      }
  }

  float2 output[32];
#pragma unroll
  for (uint dimension_tile = 0; dimension_tile < 32; ++dimension_tile)
    output[dimension_tile] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);

  const uint scale_base = page_token(fragment_row, 0);

  const SplashKvAddressing<KVHeads, int8_t> addressing(kv, kv_head);
  for (uint page = page_begin; page < page_end; ++page) {
    const auto tensors = addressing.page(page_table[page]);
    device const int8_t *keys = tensors.keys;
    device const int8_t *values = tensors.values;
    device const float *key_scales = tensors.key_scales;
    device const float *value_scales = tensors.value_scales;
    const ulong scale_index = scale_base;
    const float2 k01 = *reinterpret_cast<device const float2 *>(key_scales + scale_index);
    const float2 k45 = *reinterpret_cast<device const float2 *>(key_scales + scale_index + 4);
    const float2 v01 = *reinterpret_cast<device const float2 *>(value_scales + scale_index);
    const float2 v45 = *reinterpret_cast<device const float2 *>(value_scales + scale_index + 4);
    const float key_scale[4] = {k01.x, k01.y, k45.x, k45.y};
    const float value_scale[4] = {v01.x, v01.y, v45.x, v45.y};

    float2 scores[4];
#pragma unroll
    for (uint score_step = 0; score_step < 4; ++score_step)
      scores[score_step] = float2(0.0f);
#pragma unroll
    for (uint dimension_block = 0; dimension_block < 4; ++dimension_block)
#pragma unroll
      for (uint score_step = 0; score_step < 4; ++score_step) {
        const uint4 words = *reinterpret_cast<device const uint4 *>(
            keys + page_token(fragment_row, score_step) * HeadDimension + 64 * dimension_block + 8 * fragment_column);
#pragma unroll
        for (uint product_step = 0; product_step < 8; ++product_step)
          mma(scores[score_step], int8_pair(words[product_step >> 1] >> (8 * (product_step & 1))),
              bfloat_pair(queries[dimension_block][product_step]));
      }

    const uint token_start = page * PageTokens;
    float2 local_max(-INFINITY);
#pragma unroll
    for (uint score_step = 0; score_step < 4; ++score_step) {
      const uint token = token_start + page_token(fragment_row, score_step);
      float2 score = scores[score_step] * key_scale[score_step];
      score *= 0.0625f;
      scores[score_step] = float2(token < limit.x ? score.x : -INFINITY,
                         token < limit.y ? score.y : -INFINITY);
      local_max = max(local_max, scores[score_step]);
    }

    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));
    local_max = max(local_max, simd_shuffle_xor(local_max, 16));
    const float2 next_max = max(row_max, local_max);
    float2 probability[4];
    float2 local_sum(0.0f);
#pragma unroll
    for (uint score_step = 0; score_step < 4; ++score_step) {
      const uint token = token_start + page_token(fragment_row, score_step);
      probability[score_step] =
          float2(token < limit.x ? fast::exp(scores[score_step].x - next_max.x) : 0.0f,
                 token < limit.y ? fast::exp(scores[score_step].y - next_max.y) : 0.0f);
      local_sum += probability[score_step];
    }
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 16);
    const float2 scale(
        next_max.x == -INFINITY || next_max.x == row_max.x
            ? 1.0f : fast::exp(row_max.x - next_max.x),
        next_max.y == -INFINITY || next_max.y == row_max.y
            ? 1.0f : fast::exp(row_max.y - next_max.y));
    row_sum = row_sum * scale + local_sum;
    row_max = next_max;
    if (simd_any(scale.x != 1.0f || scale.y != 1.0f)) {
#pragma unroll
      for (uint dimension_tile = 0; dimension_tile < 32; ++dimension_tile)
        output[dimension_tile] *= scale;
    }

    float2 weights[4];
#pragma unroll
    for (uint score_step = 0; score_step < 4; ++score_step) {
      const uint token = token_start + page_token(fragment_row, score_step);
      weights[score_step] = float2(
          token < limit.x ? float(bfloat(probability[score_step].x * value_scale[score_step])) : 0.0f,
          token < limit.y ? float(bfloat(probability[score_step].y * value_scale[score_step])) : 0.0f);
    }

    device const int8_t *value_row = values + fragment_row * PageTokens + 4 * fragment_column;
#pragma unroll
    for (uint dimension_tile = 0; dimension_tile < 32; ++dimension_tile) {
      const uint2 words =
          *reinterpret_cast<device const uint2 *>(value_row + dimension_tile * 8 * PageTokens);
      mma(output[dimension_tile], int8_pair(words.x), weights[0]);
      mma(output[dimension_tile], int8_pair(words.x >> 8), weights[1]);
      mma(output[dimension_tile], int8_pair(words.y), weights[2]);
      mma(output[dimension_tile], int8_pair(words.y >> 8), weights[3]);
    }
  }

  device float *target = partials + slot * FusedRows * HeadDimension;
#pragma unroll
  for (uint dimension_tile = 0; dimension_tile < 32; ++dimension_tile) {
    target[row * HeadDimension + 8 * dimension_tile + fragment_row] = output[dimension_tile].x;
    target[(row + 1) * HeadDimension + 8 * dimension_tile + fragment_row] = output[dimension_tile].y;
  }
  if (fragment_row == 0) {
    device float *stat = statistics + (slot * FusedRows + row) * 2;
    stat[0] = row_max.x;
    stat[1] = row_sum.x;
    stat[2] = row_max.y;
    stat[3] = row_sum.y;
  }
}
}
