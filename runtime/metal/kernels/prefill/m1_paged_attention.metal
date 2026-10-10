// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "metal/kernels/common/m1_paged_attention_sgf_reuse.h"

#define M1_PREFILL_SPLIT(Name, Tile, Reuse) \
  kernel void Name( \
      device bfloat *queries [[buffer(0)]], \
      device float *partials [[buffer(1)]], \
      device float *statistics [[buffer(2)]], \
      device const SplashKvPage *page_table [[buffer(3)]], \
      constant SplashPrefillAttentionParams &params [[buffer(4)]], \
      uint3 group [[threadgroup_position_in_grid]], \
      uint simdgroup [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint Heads = 2, QueryGroup = 8; \
    const uint kv_head = group.x, query_tile = group.y, split = group.z; \
    const uint tile_start = query_tile * SplashPrefillTileRows; \
    if (!splash_prefill_attention_contract_valid(params) || \
        kv_head >= Heads || split >= params.split_count || \
        tile_start >= params.rows || simdgroup >= QueryGroup / Reuse) \
      return; \
    const uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start); \
    const ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) * \
                              QueryGroup * SplashKvHeadDimension; \
    const ulong slot = (ulong(query_tile) * Heads + kv_head) * params.split_count + split; \
    m1sgf::Tile<Heads, QueryGroup>(queries + tile_offset, page_table, params.kv, \
        kv_head, params.committed_tokens + tile_start, active_rows, \
        params.split_count, split, partials, statistics, slot, simdgroup, lane); \
  }

M1_PREFILL_SPLIT(prefill_attention_q8_m1_sgf_kv2_g8, tile, 1)
M1_PREFILL_SPLIT(prefill_attention_q8_m1_reuse2_kv2_g8, tile_reuse2, 2)
#undef M1_PREFILL_SPLIT
