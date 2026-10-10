// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDecodeRows = SPLASH_TARGET_VERIFY_ROWS;
static_assert(kMaximumLanes == 4);

uint32_t m1PrefillMoeM8MaxRows(const DeviceCapabilities &device) {
  if (!DeviceCapabilities::isM1Ultra(device.hardwareAppleGpuFamily, device.gpuCoreCount))
    return 0;
  const char *value = std::getenv("SPLASH_M1_PREFILL_MOE_M8_MAX_ROWS");
  if (!value || !*value)
    return 0;
  const std::string_view text(value);
  uint32_t rows = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), rows);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || rows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("SPLASH_M1_PREFILL_MOE_M8_MAX_ROWS must be an integer from 0 to 2048");
  return rows;
}

bool m1AttentionFlag(const DeviceCapabilities &device, const char *name, bool fallback) {
  if (!DeviceCapabilities::isM1Ultra(device.hardwareAppleGpuFamily, device.gpuCoreCount))
    return false;
  const char *value = std::getenv(name);
  if (!value || !*value) return fallback;
  const std::string_view text(value);
  if (text == "0") return false;
  if (text == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

template <typename Workspace, size_t N>
void include(Workspace &bound, const Workspace &required,
             const std::array<uint64_t Workspace::*, N> &fields,
             uint32_t lanes = 1) {
  for (auto field : fields) {
    const uint64_t bytes = required.*field;
    bound.*field = std::max(bound.*field,
                           bytes / lanes + uint64_t{bytes % lanes != 0});
  }
}

} // namespace

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device)
    : linear_(device), moeRouteWideRows_(moeRouteWideRows(plannedGpuCores(device))),
      family_(gpuFamilyClass(device.appleGpuFamily)),
      m1Ultra_(DeviceCapabilities::isM1Ultra(device.hardwareAppleGpuFamily, device.gpuCoreCount)),
      m1PrefillMoeM8MaxRows_(m1PrefillMoeM8MaxRows(device)),
      m1AttentionRegister_(m1AttentionFlag(device, "SPLASH_M1_ATTN_SGF", true)),
      m1AttentionReuse2_(m1AttentionFlag(device, "SPLASH_M1_PREFILL_ATTN_REUSE2", false)) {}

PrefillAttentionPlan ExecutionPlans::prefillAttention(
    uint32_t rows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillPlan(rows, queryHeads, layout, m1AttentionRegister_, m1AttentionReuse2_);
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens) const {
  return PagedAttention::verifyPlan(lanes, queryHeads, layout, historyTokens, m1AttentionRegister_);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes) const {
  return DraftAttention::plan(shape, lanes);
}

// The router threshold, the expert tile, the simdgroups of a decode plan's
// 8-row tiles and, for a GGUF plan, its tiles.
MoeConfig ExecutionPlans::moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const {
  const bool prefill = phase == MoePhase::Prefill;
  MoeConfig config;
  config.routeWideRows = moeRouteWideRows_;
  config.expertTile = prefill ? MoeExpertTile::M32 : MoeExpertTile::M8;
  if (m1Ultra_ && shape.weightLayout == WeightLayout::Affine64)
    config.kernel = MoeExpertKernel::M1Register;
  if (!prefill) config.m8Simdgroups = moeDecodeSimdgroups(family_);
  if (prefill && shape.weightLayout == WeightLayout::Affine64 && m1Ultra_ && rows <= m1PrefillMoeM8MaxRows_)
    config.expertTile = MoeExpertTile::M8;
  if (shape.weightLayout == WeightLayout::Block32) {
    const MoeGgufTile tile = moeGgufTile(family_, shape);
    if (prefill) config.expertTile = moeGgufPrefillTile(shape, rows, tile);
    config.ggufTile = tile;
    config.ggufRouterTile = linear_.ggufFloatTile(rows, shape.experts);
  }
  return config;
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t rows) const {
  return MoE::prefillPlan(shape, rows, moeConfig(shape, rows, MoePhase::Prefill));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t lanes) const {
  // Validate before multiplying an untrusted width into the plan's rows.
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid MoE decode width");
  return MoE::decodePlan(shape, lanes, moeConfig(shape, lanes * kDecodeRows, MoePhase::Decode));
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
}

// The verify bound is linear in the lanes: one lane's is every width's share.
AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerLane(
    uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::verifyWorkspace(1, queryHeads, layout);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape) const {
  return DraftAttention::plan(shape, 1).workspace();
}

MoeWorkspace ExecutionPlans::moePrefillWorkspace(MoeShape shape,
                                               uint32_t maximumRows) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  auto bound = moePrefill(shape, maximumRows).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows)
    include(bound, moePrefill(shape, rows).workspace(), kMoeWorkspaceFields);
  return bound;
}

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerLane(MoeShape shape) const {
  MoeWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, moeDecode(shape, lanes).workspace(), kMoeWorkspaceFields, lanes);
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, lanes * kDecodeRows,
                                  LinearPhase::Decode, LinearEpilogue::GateUp, shape.layout};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops
