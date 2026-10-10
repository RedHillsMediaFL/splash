// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace splash::model {

class PromptLookup final {
public:
  void reset(std::span<const uint32_t> prompt) {
    prompt_.assign(prompt.begin(), prompt.end());
    heads_.assign(kBuckets, kMissing);
    previous_.assign(prompt.size(), kMissing);
    for (uint32_t end = kIndexTokens; end < prompt.size(); ++end) {
      const uint32_t bucket = hash(prompt.subspan(end - kIndexTokens, kIndexTokens));
      previous_[end] = heads_[bucket];
      heads_[bucket] = end;
    }
    historySize_ = 0;
    observe(prompt.last(std::min(prompt.size(), history_.size())));
    cooldown_ = 0;
    hint_ = kMissing;
  }

  void observe(std::span<const uint32_t> tokens) {
    for (const uint32_t token : tokens) {
      if (historySize_ == history_.size()) {
        std::move(history_.begin() + 1, history_.end(), history_.begin());
        --historySize_;
      }
      history_[historySize_++] = token;
    }
  }

  bool propose(uint32_t anchor, std::span<uint32_t> output) {
    if (cooldown_) {
      --cooldown_;
      return false;
    }
    if (heads_.empty() || output.empty() || historySize_ + 1 < kMinimumMatch)
      return false;
    std::array<uint32_t, kHistory + 1> suffix{};
    std::copy_n(history_.begin(), historySize_, suffix.begin());
    suffix[historySize_] = anchor;
    const uint32_t count = historySize_ + 1;
    const uint32_t bucket = hash(std::span(suffix).subspan(count - kIndexTokens, kIndexTokens));
    uint32_t bestEnd = kMissing;
    uint32_t bestMatch = kMinimumMatch - 1;
    auto matchLength = [&](uint32_t end) {
      uint32_t match = 0;
      while (match < count && match < end &&
             suffix[count - match - 1] == prompt_[end - match - 1])
        ++match;
      return match;
    };
    if (hint_ != kMissing && hint_ < prompt_.size() &&
        output.size() <= prompt_.size() - hint_) {
      const uint32_t match = matchLength(hint_);
      if (match > bestMatch) {
        bestEnd = hint_;
        bestMatch = match;
      }
    }
    uint32_t end = heads_[bucket];
    for (uint32_t probes = 0; end != kMissing && probes < 32; ++probes) {
      if (output.size() <= prompt_.size() - end) {
        const uint32_t match = matchLength(end);
        if (match > bestMatch) {
          bestMatch = match;
          bestEnd = end;
          if (match == count)
            break;
        }
      }
      end = previous_[end];
    }
    if (bestEnd == kMissing)
      return false;
    lastEnd_ = bestEnd;
    std::copy_n(prompt_.begin() + bestEnd, output.size(), output.begin());
    return true;
  }

  void feedback(uint32_t accepted) {
    cooldown_ = accepted < 3 ? 4 : 0;
    hint_ = lastEnd_ + accepted + 1;
  }

private:
  static constexpr uint32_t kBuckets = 1U << 16;
  static constexpr uint32_t kMissing = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kIndexTokens = 4;
  static constexpr uint32_t kMinimumMatch = 8;
  static constexpr uint32_t kHistory = 16;
  static uint32_t hash(std::span<const uint32_t> tokens) {
    uint32_t value = 2166136261U;
    for (uint32_t token : tokens)
      value = (value ^ token) * 16777619U;
    return (value ^ (value >> 16)) & (kBuckets - 1);
  }
  std::vector<uint32_t> prompt_, heads_, previous_;
  std::array<uint32_t, kHistory> history_{};
  uint32_t historySize_ = 0;
  uint32_t cooldown_ = 0;
  uint32_t hint_ = kMissing, lastEnd_ = 0;
};

}
