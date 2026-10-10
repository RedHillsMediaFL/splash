// Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
#include "model/PromptLookup.hpp"

#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using splash::model::PromptLookup;

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

int main() {
  std::vector<uint32_t> prompt(262144);
  std::iota(prompt.begin(), prompt.end(), 0);
  PromptLookup lookup;
  lookup.reset(prompt);
  std::array<uint32_t, 7> proposed{};
  require(!lookup.propose(42, proposed), "unrelated suffix must miss");
  lookup.observe(std::array<uint32_t, 8>{896, 897, 898, 899, 900, 901, 902, 903});
  require(lookup.propose(904, proposed), "matching suffix must propose");
  for (uint32_t position = 0; position < proposed.size(); ++position)
    require(proposed[position] == 905 + position, "proposal must follow match");
  lookup.feedback(0);
  for (uint32_t attempt = 0; attempt < 4; ++attempt)
    require(!lookup.propose(904, proposed), "rejection cooldown missing");
  require(lookup.propose(904, proposed), "cooldown should expire");
  lookup.feedback(7);
  require(lookup.propose(904, proposed), "accepted proposal should stay enabled");
  lookup.reset(std::array<uint32_t, 3>{900, 901, 902});
  require(!lookup.propose(903, proposed), "short prompt must not overrun");
  lookup.reset(prompt);
  lookup.observe(std::span(prompt).last(3));
  require(!lookup.propose(prompt.back(), proposed), "end of prompt must not overrun");
  lookup.reset(std::array<uint32_t, 32>{
      90, 91, 92, 93, 94, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16,
      80, 81, 82, 83, 84, 1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 26});
  lookup.observe(std::array<uint32_t, 8>{90, 91, 92, 93, 94, 1, 2, 3});
  require(lookup.propose(4, proposed) && proposed[0] == 10,
          "longest matching continuation should win");
  std::vector<uint32_t> repeated(120);
  std::iota(repeated.begin(), repeated.end(), 1000);
  std::copy_n(repeated.begin() + 8, 30, repeated.begin() + 68);
  lookup.reset(repeated);
  lookup.observe(std::span(repeated).first(8));
  require(lookup.propose(repeated[8], proposed) && proposed[0] == repeated[9],
          "initial unique occurrence must match");
  for (uint32_t anchor = 16; anchor <= 32; anchor += 8) {
    lookup.feedback(7);
    lookup.observe(std::span(repeated).subspan(anchor - 8, 8));
    require(lookup.propose(repeated[anchor], proposed), "verified occurrence lost");
    for (uint32_t position = 0; position < proposed.size(); ++position)
      require(proposed[position] == repeated[anchor + position + 1],
              "verified occurrence must win an ambiguous suffix tie");
  }
  std::fill(prompt.begin(), prompt.end(), 7);
  lookup.reset(prompt);
  const auto start = std::chrono::steady_clock::now();
  for (uint32_t attempt = 0; attempt < 10000; ++attempt)
    require(lookup.propose(7, proposed), "repeated prompt should match");
  require(std::chrono::steady_clock::now() - start < std::chrono::seconds(2),
          "lookup work should be bounded");
  lookup.reset({});
  require(!lookup.propose(7, proposed), "reset must clear previous request data");
  std::cout << "prompt_lookup_test: PASS\n";
}
