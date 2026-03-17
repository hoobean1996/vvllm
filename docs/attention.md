# Attention
Q, K, V all come from input.

input: [seqlen, 896]

1. Projection
Q = input * q
K = input * k
V = input * v

2. Attention
S(scores) = Q * K^T
S = scores / sqrt(64)
scores = softmax(scores)
attn = scores x V

3. Output
O(output) = attn x o

4. Residual connection
output = input + ouput

5. MLP
6. Residual connection

Repeat steps 1-6 for all 24 layers, then 

7. Final norm + predict


Notes:
- Each layer has its own q, k, v weights config.

- Training (Qwen team did this) text data → training loop → learned weights → saved to model.safetensors

- Inference (what we're building) model.safetensors → load weights → multiply with input → get output
