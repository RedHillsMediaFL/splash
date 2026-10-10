// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "model/Model.hpp"
#include "ops/Embedding.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::Embedding;
using splash::ops::RowCopy;
using splash::ops::RowRegion;

constexpr uint32_t kRows = splash::model::ExecutionLimits::targetVerifyRows;
constexpr uint32_t kProposals =
    splash::model::ExecutionLimits::draftProposalTokens;
constexpr uint32_t kLanes =
    splash::model::ExecutionLimits::maximumBatchWidth;

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

template <class T> T *contents(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

using splash::test::rejects;
using splash::test::require;
using splash::test::requireExtent;

void runWidth(MetalBackend &backend, uint32_t width,
              const std::array<uint32_t, kLanes> &acceptedReference,
              const std::array<uint32_t, kLanes> &remainingReference,
              uint32_t stopLane = kLanes, uint32_t stopRow = 0) {
  require(width >= 1 && width <= kLanes, "invalid test width");
  MetalBuffer draft = shared(backend, kLanes * kProposals * sizeof(uint32_t),
                             "accept-draft");
  MetalBuffer draftIds =
      shared(backend, kLanes * kProposals * 16 * sizeof(uint32_t),
             "accept-draft-ids");
  MetalBuffer draftProbabilities =
      shared(backend, kLanes * kProposals * 16 * sizeof(float),
             "accept-draft-probabilities");
  // Every lane is greedy here; sampled lanes read their target rows.
  MetalBuffer targetRows =
      shared(backend, kLanes * kRows * sizeof(TargetVocabularyRow),
             "accept-target-rows");
  MetalBuffer uniforms =
      shared(backend, kLanes * SPLASH_SAMPLING_UNIFORMS * sizeof(float),
             "accept-uniforms");
  MetalBuffer output = shared(backend, kLanes * kRows * sizeof(uint32_t),
                              "accept-output");
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "accept-retained");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "accept-count");

  auto *draftTokens = contents<uint32_t>(draft);
  auto *targetTokens = contents<uint32_t>(output);
  std::memset(draftIds.contents(), 0, draftIds.sizeBytes());
  std::memset(draftProbabilities.contents(), 0,
              draftProbabilities.sizeBytes());
  std::memset(targetRows.contents(), 0, targetRows.sizeBytes());
  std::memset(uniforms.contents(), 0, uniforms.sizeBytes());
  std::memset(retained.contents(), 0, retained.sizeBytes());
  std::memset(accepted.contents(), 0, accepted.sizeBytes());
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    for (uint32_t token = 0; token < kProposals; ++token) {
      const uint32_t proposal = 1000 + lane * 100 + token;
      draftTokens[lane * kProposals + token] = proposal;
      targetTokens[lane * kRows + token] =
          token < acceptedReference[lane] ? proposal : proposal + 50;
    }
    targetTokens[lane * kRows + kRows - 1] = 9000 + lane;
  }

  constexpr uint32_t kStopToken = 248044;
  if (stopLane < width) {
    require(stopRow < kRows, "invalid stop row");
    targetTokens[stopLane * kRows + stopRow] = kStopToken;
    if (stopRow < acceptedReference[stopLane])
      draftTokens[stopLane * kProposals + stopRow] = kStopToken;
  }

  AcceptBatchParams params{};
  std::copy(remainingReference.begin(), remainingReference.end(),
            std::begin(params.remaining));
  params.stop_token_0 = kStopToken;
  params.stop_token_1 = 248046;
  ComputeDispatch dispatch;
  dispatch.pipelineName = "decode_accept_dflash";
  dispatch.buffers = {{0, draft},
                      {1, draftIds},
                      {2, draftProbabilities},
                      {3, targetRows},
                      {4, uniforms},
                      {5, output},
                      {6, retained},
                      {7, accepted}};
  dispatch.bytes = {{8, &params, sizeof(params)}};
  dispatch.threadgroups = {width, 1, 1};
  dispatch.threadsPerThreadgroup = {1, 1, 1};
  static_cast<void>(backend.submit(dispatch));

  const auto *retainedCounts = contents<uint32_t>(retained);
  const auto *acceptedCounts = contents<uint32_t>(accepted);
  for (uint32_t lane = 0; lane < width; ++lane) {
    uint32_t expectedRetained =
        std::min(acceptedReference[lane] + 1, remainingReference[lane]);
    if (lane == stopLane && stopRow < expectedRetained)
      expectedRetained = stopRow + 1;
    require(acceptedCounts[lane] == acceptedReference[lane],
            "accepted proposal count mismatch");
    require(retainedCounts[lane] == expectedRetained,
            "retained token count mismatch");
  }
}

// The row copies of production, bitwise against a CPU copy: the capture of
// 13 target hidden rows (width 2048) from source row 5 into the third slot of
// four of the captured rows from row 2, the gather of the last 3 of 11
// prefill rows into the head's input, and an image's embedding rows 1-2
// (width 5120) over a chunk's placeholder rows from row 2. Every value
// outside the destination region keeps its poison, and regions past a row are
// refused. Each buffer holds its region at its extent, the end of its last
// row's values, and is refused one value short.
void testFusedGreedy(MetalBackend &backend, uint32_t width,
                     const std::array<uint32_t, kLanes> &acceptedReference,
                     const std::array<uint32_t, kLanes> &remainingReference,
                     uint32_t stopLane = kLanes, uint32_t stopRow = 0,
                     bool excludesStops = false, bool invalidRow = false) {
  constexpr uint32_t vocabulary = 128;
  constexpr uint32_t stopToken = 126;
  const uint32_t rows = width * kRows;
  const splash::ops::Sampling sampling(vocabulary);
  const auto workspace = splash::ops::Sampling::workspace(rows);
  splash::ops::SamplingBuffers buffers;
  buffers.logits = shared(backend, uint64_t{rows} * vocabulary * sizeof(float), "fused-logits");
  buffers.argmaxValues = shared(backend, workspace.argmaxValuesBytes, "fused-values");
  buffers.argmaxIndices = shared(backend, workspace.argmaxIndicesBytes, "fused-indices");
  buffers.outputTokens = shared(backend, rows * sizeof(uint32_t), "fused-output");
  buffers.constraintMasks = shared(backend, width * (kRows + 1) * 16, "fused-masks");
  splash::ops::AcceptanceBuffers acceptance;
  acceptance.proposedTokens = shared(backend, width * kProposals * sizeof(uint32_t), "fused-proposals");
  acceptance.candidates = shared(backend, width * kProposals * SPLASH_DRAFT_CANDIDATES * sizeof(uint32_t), "fused-candidates");
  acceptance.proposalProbabilities = shared(backend, width * kProposals * SPLASH_DRAFT_CANDIDATES * sizeof(float), "fused-probabilities");
  acceptance.targetVocabularyRows = shared(backend, workspace.vocabularyRowsBytes, "fused-target-rows");
  acceptance.uniforms = shared(backend, width * SPLASH_SAMPLING_UNIFORMS * sizeof(float), "fused-uniforms");
  acceptance.outputTokens = buffers.outputTokens;
  acceptance.retainedCounts = shared(backend, width * sizeof(uint32_t), "fused-retained");
  acceptance.acceptedCounts = shared(backend, width * sizeof(uint32_t), "fused-accepted");
  auto *logits = contents<float>(buffers.logits);
  std::fill_n(logits, rows * vocabulary, -1.0F);
  std::array<splash::ops::SamplingPolicy, kLanes> policies{};
  auto *draft = contents<uint32_t>(acceptance.proposedTokens);
  for (uint32_t lane = 0; lane < width; ++lane) {
    policies[lane].excludesStopTokens = excludesStops;
    for (uint32_t row = 0; row < kRows; ++row) {
      const uint32_t proposal = 20 + lane * 12 + row;
      uint32_t target = row < acceptedReference[lane] ? proposal : 80 + row;
      if (lane == stopLane && row == stopRow)
        target = stopToken;
      if (row < kProposals)
        draft[lane * kProposals + row] = row < acceptedReference[lane] ? target : proposal;
      float *rowLogits = logits + (lane * kRows + row) * vocabulary;
      rowLogits[target] = 2.0F;
      if (target < 110)
        rowLogits[110] = 2.0F;
    }
  }
  if (invalidRow)
    std::fill_n(logits, vocabulary, std::numeric_limits<float>::quiet_NaN());
  const auto activePolicies = std::span(policies).first(width);
  const auto maximumRetained = std::span(remainingReference).first(width);
  CommandGraph reference;
  sampling.addVerify(reference, activePolicies, buffers, stopToken, 127, {});
  sampling.addAcceptance(reference, acceptance, maximumRetained, activePolicies, stopToken, 127);
  static_cast<void>(backend.submitCommandAsync(reference.dispatches()).wait());
  const auto *referenceOutput = contents<uint32_t>(buffers.outputTokens);
  const std::vector<uint32_t> output(referenceOutput, referenceOutput + rows);
  const auto *referenceRetained = contents<uint32_t>(acceptance.retainedCounts);
  const std::vector<uint32_t> retained(referenceRetained, referenceRetained + width);
  const auto *referenceAccepted = contents<uint32_t>(acceptance.acceptedCounts);
  const std::vector<uint32_t> accepted(referenceAccepted, referenceAccepted + width);
  std::fill_n(contents<uint32_t>(buffers.outputTokens), rows, 0xDEADBEEFu);
  std::fill_n(contents<uint32_t>(acceptance.retainedCounts), width, 0xDEADBEEFu);
  std::fill_n(contents<uint32_t>(acceptance.acceptedCounts), width, 0xDEADBEEFu);
  CommandGraph fused;
  sampling.addGreedyVerifyAndAcceptance(fused, buffers, acceptance, maximumRetained,
                                        activePolicies, stopToken, 127);
  require(reference.dispatches().size() == 3 && fused.dispatches().size() == 2,
          "greedy fusion must remove one dispatch");
  static_cast<void>(backend.submitCommandAsync(fused.dispatches()).wait());
  require(std::equal(output.begin(), output.end(), contents<uint32_t>(buffers.outputTokens)),
          "fused greedy output differs from ordinary policy");
  require(std::equal(retained.begin(), retained.end(), contents<uint32_t>(acceptance.retainedCounts)),
          "fused greedy retention differs from ordinary policy");
  require(std::equal(accepted.begin(), accepted.end(), contents<uint32_t>(acceptance.acceptedCounts)),
          "fused greedy acceptance differs from ordinary policy");
  for (const uint32_t mutation : {0u, 1u, 2u}) {
    auto invalidPolicies = policies;
    if (mutation == 0)
      invalidPolicies[0].temperature = 1.0F;
    else if (mutation == 1)
      invalidPolicies[0].constrained = true;
    else
      invalidPolicies[0].penalties.repetition = 1.2F;
    CommandGraph invalid;
    rejects([&] { sampling.addGreedyVerifyAndAcceptance(invalid, buffers, acceptance, maximumRetained,
                                                       std::span(invalidPolicies).first(width), stopToken, 127); },
            "invalid fused greedy lane", "unsupported fused policy must be refused");
    require(invalid.dispatches().empty(), "invalid fused policy encoded a partial graph");
  }
  requireExtent(backend, acceptance.outputTokens, rows * sizeof(uint32_t), sizeof(uint32_t),
                "target token", [&](CommandGraph &graph, const MetalBuffer &view) {
                  auto changed = acceptance;
                  changed.outputTokens = view;
                  sampling.addGreedyVerifyAndAcceptance(graph, buffers, changed, maximumRetained,
                                                        activePolicies, stopToken, 127);
                });
}

void testRowCopy(MetalBackend &backend) {
  constexpr uint16_t kPoison = 0xA5A5;
  const auto check = [&](uint32_t sourceRows, RowRegion from,
                         uint32_t destinationRows, RowRegion to,
                         uint32_t rows, uint32_t width, const char *what) {
    MetalBuffer source = shared(backend, uint64_t{sourceRows} * from.stride * 2, "row-copy-source");
    MetalBuffer destination =
        shared(backend, uint64_t{destinationRows} * to.stride * 2, "row-copy-destination");
    auto *input = contents<uint16_t>(source);
    for (uint64_t i = 0; i < source.sizeBytes() / 2; ++i)
      input[i] = static_cast<uint16_t>(i * 2654435761u >> 16);
    std::vector<uint16_t> expected(destination.sizeBytes() / 2, kPoison);
    std::copy(expected.begin(), expected.end(), contents<uint16_t>(destination));
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t column = 0; column < width; ++column)
        expected[uint64_t{to.row + row} * to.stride + to.column + column] =
            input[uint64_t{from.row + row} * from.stride + from.column + column];
    CommandGraph graph;
    RowCopy::add(graph, source, from, destination, to, rows, width);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    require(std::equal(expected.begin(), expected.end(), contents<uint16_t>(destination)), what);
  };
  constexpr uint32_t kWidth = 2048;
  check(18, {5, kWidth, 0}, 16, {2, 4 * kWidth, 2 * kWidth}, 13, kWidth,
        "captured rows differ from a CPU copy");
  check(11, {8, kWidth, 0}, kRows, {0, kWidth, 0}, 3, kWidth,
        "gathered rows differ from a CPU copy");
  constexpr uint32_t kImageWidth = 5120;
  check(4, {1, kImageWidth, 0}, 5, {2, kImageWidth, 0}, 2, kImageWidth,
        "image rows differ from a CPU copy");

  MetalBuffer rows = shared(backend, uint64_t{4} * kWidth * 2, "row-copy-invalid");
  CommandGraph invalid;
  const auto copy = [&](RowRegion from, RowRegion to, uint32_t count, uint32_t width) {
    RowCopy::add(invalid, rows, from, rows, to, count, width);
  };
  rejects([&] { copy({0, kWidth, 0}, {0, kWidth, 0}, 0, kWidth); }, "invalid row copy",
          "a row copy of no rows was accepted");
  rejects([&] { copy({0, kWidth, 0}, {0, kWidth, 0}, 1, 0); }, "invalid row copy",
          "a row copy of no values was accepted");
  rejects([&] { copy({0, kWidth, 1}, {0, kWidth, 0}, 1, kWidth); }, "invalid row copy",
          "a row copy past the end of its source row was accepted");
  rejects([&] { copy({0, kWidth, 0}, {0, kWidth / 2, 0}, 1, kWidth); }, "invalid row copy",
          "a row copy past the end of its destination row was accepted");
  require(invalid.empty(), "an invalid row copy encoded a dispatch");
  // The capture's regions: source rows 5-17 of 2048 values, destination rows
  // 2-14 of 8192 from column 4096.
  constexpr RowRegion from{5, kWidth, 0}, to{2, 4 * kWidth, 2 * kWidth};
  constexpr uint32_t copied = 13;
  const MetalBuffer source = shared(backend, uint64_t{18} * kWidth * 2, "row-copy-source"),
                    destination = shared(backend, uint64_t{16} * 4 * kWidth * 2, "row-copy-destination");
  requireExtent(backend, source, uint64_t{17 * kWidth + kWidth} * 2, 2, "row copy source",
                [&](CommandGraph &graph, const MetalBuffer &view) {
                  RowCopy::add(graph, view, from, destination, to, copied, kWidth);
                });
  requireExtent(backend, destination, (uint64_t{14} * 4 * kWidth + 2 * kWidth + kWidth) * 2, 2,
                "row copy destination", [&](CommandGraph &graph, const MetalBuffer &view) {
                  RowCopy::add(graph, source, from, view, to, copied, kWidth);
                });
}

// Each buffer the verify input reaches, at its extent and one element short,
// for three lanes: each lane's anchor, row 0 of its eight draft input rows,
// its seven proposals and its eight verify input rows.
void verifyInputExtents(MetalBackend &backend) {
  constexpr uint32_t lanes = 3, vocabulary = 1003;
  const std::array buffers{shared(backend, lanes * kRows * 4, "draft-input"),
                           shared(backend, lanes * kProposals * 4, "proposed"),
                           shared(backend, lanes * kRows * 4, "verify-input")};
  for (const auto &[index, bytes, what] :
       {std::tuple{size_t{0}, uint64_t{(lanes - 1) * kRows + 1} * 4, "draft input token"},
        std::tuple{size_t{1}, uint64_t{lanes * kProposals} * 4, "proposed token"},
        std::tuple{size_t{2}, uint64_t{lanes * kRows} * 4, "verify input token"}})
    requireExtent(backend, buffers[index], bytes, 4, what, [&](CommandGraph &graph, const MetalBuffer &view) {
      auto changed = buffers;
      changed[index] = view;
      Embedding::addVerifyInput(graph, changed[0], changed[1], changed[2], vocabulary, lanes);
    });
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: dflash-batch-control METALLIB");
    MetalBackend backend(argv[1]);
    constexpr std::array<uint32_t, kLanes> lowerAccepted{0, 1, 2, 3};
    constexpr std::array<uint32_t, kLanes> upperAccepted{4, 5, 6, 7};
    constexpr std::array<uint32_t, kLanes> fullRemaining{8, 8, 8, 8};
    for (uint32_t width = 1; width <= kLanes; ++width) {
      runWidth(backend, width, lowerAccepted, fullRemaining);
      runWidth(backend, width, upperAccepted, fullRemaining);
      testFusedGreedy(backend, width, lowerAccepted, fullRemaining);
      testFusedGreedy(backend, width, upperAccepted, fullRemaining);
      testFusedGreedy(backend, width, upperAccepted, fullRemaining, kLanes, 0, false, true);
    }

    // Output limits and stop tokens shorten the committed prefix without
    // changing the physical eight-row graph.  The accepted proposal count is
    // still seven in every lane; only retained rows move.
    constexpr std::array<uint32_t, kLanes> allAccepted{7, 7, 7, 7};
    constexpr std::array<uint32_t, kLanes> shortRemaining{1, 2, 3, 8};
    runWidth(backend, kLanes, allAccepted, shortRemaining, 3, 3);
    testFusedGreedy(backend, kLanes, allAccepted, shortRemaining, 3, 3);
    testFusedGreedy(backend, kLanes, allAccepted, shortRemaining, 3, 3, true);
    testRowCopy(backend);
    verifyInputExtents(backend);
    std::cout << "dflash_batch_control_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dflash_batch_control_metal_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
