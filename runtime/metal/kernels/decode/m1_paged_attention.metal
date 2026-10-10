// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/m1_paged_attention_sgf.h"

kernel void verify_attention_q8_m1_sgf_kv2_g8(
    device bfloat *queries [[buffer(0)]],
    device float *partials [[buffer(1)]],
    device float *statistics [[buffer(2)]],
    device const SplashKvPage *page_table0 [[buffer(3)]],
    device const SplashKvPage *page_table1 [[buffer(4)]],
    device const SplashKvPage *page_table2 [[buffer(5)]],
    device const SplashKvPage *page_table3 [[buffer(6)]],
    constant SplashVerifyAttentionParams *params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint Heads = 2, QueryGroup = 8;
  const uint kv_head = group.x, split = group.y, batch = group.z;
  if (batch >= SPLASH_MAXIMUM_BATCH_WIDTH || kv_head >= Heads || simdgroup >= QueryGroup)
    return;
  constant SplashVerifyAttentionParams &lane_params = params[batch];
  if (!splash_verify_attention_contract_valid(lane_params) || split >= lane_params.split_count)
    return;
  constexpr ulong group_stride = ulong(SPLASH_VERIFY_CHUNK_STRIDE) * QueryGroup * SplashKvHeadDimension;
  device const SplashKvPage *page_table = SPLASH_LANE_BINDING(batch, page_table0, page_table1, page_table2, page_table3);
  const ulong slot = (ulong(batch) * Heads + kv_head) * lane_params.slot_splits + split;
  m1sgf::tile<Heads, QueryGroup>(queries + (ulong(batch) * Heads + kv_head) * group_stride,
      page_table, lane_params.kv, kv_head, lane_params.committed_tokens,
      SPLASH_TARGET_VERIFY_ROWS, lane_params.split_count, split,
      partials, statistics, slot, simdgroup, lane);
}
