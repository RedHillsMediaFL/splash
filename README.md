# Splash for M1 Ultra

Modified by Red Hills Media on 2026-10-10.

A focused source fork of [incoai/Splash](https://github.com/incoai/splash), based
on upstream commit `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`.
The `m1-ultra` branch contains our tested M1 Ultra engine changes; upstream
`main` remains separate. This release contains source only, not binaries or
model weights.

## Hardware and changes

Validated on an **M1 Ultra, 48-core GPU, 64 GB unified memory**.
M1 Ultra detection also recognizes the 64-core part, which is not a separate
benchmark result. Smaller M1/M2 systems are not newly supported. Existing
upstream model families and newer-GPU paths remain unchanged.

- Physical Apple7 admission with M1-specific affine-Q4 projection and MoE kernels.
- Register-based INT8 paged attention for prefill, decode, and verification.
- Verified prompt lookup and fused greedy sampling, with runtime telemetry.
- GPU-only prefill FFNs on M1 Ultra; its unbounded ANE calibration path is avoided.
- Request-local `none/low/medium/high/xhigh/max` effort profiles, with ceilings
  of 0/64/256/1024/4096/16384 reasoning tokens and visible-answer reservation.
- Existing-family MLX checkpoint staging and tool/API compatibility fixes.

Effort levels are runtime instructions and enforced ceilings, not a claim that
the weights were trained for six levels or that higher effort improves accuracy.
See [DEVELOPMENT.md](DEVELOPMENT.md) for API details and upstream limitations.

## Build from source

Requires macOS 26.4+, Xcode 26+ with the newer Metal 4 tensor compiler, and
Python 3.12–3.14. The upstream Homebrew binary is not this fork.

```sh
git clone --branch m1-ultra https://github.com/RedHillsMediaFL/splash.git
cd splash
make -j4
SPLASH_M1_PREFILL_MOE_M8_MAX_ROWS=2048 SPLASH_PROMPT_LOOKUP=1 SPLASH_FUSED_GREEDY=1 \
  ./splash serve --model redhillsmediafl/Longleaf-35B-A3B \
  --language-only --disable-ane --max-memory 28G --max-context 262144 \
  --default-reasoning-effort none
```

The first serve downloads the target and its matching draft and prepares Python
dependencies. The 28 GiB setting caps tracked native allocations, not whole
machine memory; leave host headroom.

## One warm coding smoke run

Historical Pi repair with Longleaf and medium effort on the hardware above:

| Model calls | Prompt tokens per call | Output tokens | Decode | First token |
| --- | --- | --- | --- | --- |
| 4 | 1,731–2,236 | 439 total | 104–130 tokens/s | 1.3 s first; 0.2–0.3 s cached |

Read/edit/test completed in about **6 seconds**, with **4/4 tests passing**.
The model was already loaded. This is a small cached smoke run, not a standard
benchmark, a baseline speedup, or a speed guarantee.

## Source versus deployed binary

The measured native source identity was
`bf1ac87654060b0cd9e807460756e8adc61a6dea5da14848f4039066b83eeb3b`;
its executable SHA256 was
`8e26a91e9da3176711e0527d2e9dbb8482b4a506eef001f2d025f4c0ee5f6ed4`.
The source was checked against that identity before adding public modification
notices and whitespace-only cleanup. These change source fingerprints. No binary is distributed or
rebuilt for this source release, and byte-for-byte build reproducibility is not
claimed.

## License

[Apache 2.0](LICENSE), with existing upstream attribution and
[third-party notices](THIRD_PARTY_NOTICES) retained.
