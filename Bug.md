# RoPE Pairing Convention Bug

## Symptom

Model output was wrong. For prompt "The capital of France is", the model predicted garbage ("5ĊAĊAnswered") instead of "Paris". The token "Paris" had a logit of only 5.82 instead of the correct 17.84.

## Root Cause

RoPE (Rotary Position Embeddings) works by rotating pairs of elements in each attention head. There are two conventions for which elements get paired:

**Interleaved (GPT-NeoX style)** — what we had (wrong for Qwen2):
```
head = [x0, x1, x2, x3, x4, x5, ...]
pairs: (x0, x1), (x2, x3), (x4, x5), ...
         ↑   ↑
       adjacent elements
```

**Split-half (LLaMA/HuggingFace style)** — what Qwen2 actually uses:
```
head = [x0, x1, x2, ..., x31, x32, x33, x34, ..., x63]
        |___________________________|
pairs: (x0, x32), (x1, x33), (x2, x34), ...
         ↑                     ↑
       first half           second half
```

Both conventions apply the same rotation math — `cos/sin` with the same frequencies — but to **different element pairs**. The rotation itself is identical:

```
new_a = a * cos(θ) - b * sin(θ)
new_b = a * sin(θ) + b * cos(θ)
```

The difference is only which `(a, b)` get paired.

With the wrong pairing, every attention score was computed with incorrectly rotated Q and K vectors, so the model couldn't attend to the right tokens — producing garbage output.

## Fix

In `src/backend/backend_naive.cc`, changed the indexing from interleaved to split-half:

```cpp
// Before (interleaved — wrong for Qwen2):
float x0 = head[2 * i];
float x1 = head[2 * i + 1];
head[2 * i]     = x0 * cos_a - x1 * sin_a;
head[2 * i + 1] = x0 * sin_a + x1 * cos_a;

// After (split-half — correct for Qwen2):
std::size_t half = head_dim / 2;
float x0 = head[i];
float x1 = head[i + half];
head[i]        = x0 * cos_a - x1 * sin_a;
head[i + half] = x0 * sin_a + x1 * cos_a;
```

This matches HuggingFace's `rotate_half()` function, which splits the vector in two halves and rotates across them.

## How It Was Debugged

1. Observed wrong output, added debug prints for top logits
2. Verified BF16 weight loading — matched Python reference exactly
3. Verified Q projection — matched Python reference exactly
4. Ran full numpy reference forward pass — all intermediate values matched C++
5. This meant the algorithm was consistent between C++ and Python, but both used the wrong RoPE convention
6. Tested both RoPE conventions in Python against HuggingFace transformers
7. Split-half matched HuggingFace exactly; interleaved did not
8. Fixed C++ implementation to use split-half
