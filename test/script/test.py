import torch
from einops import rearrange
import xformers.ops

# 你的 efficient_attn 函数
def efficient_attn(q, k, v, num_heads, b):
    X = q.shape[0] // (num_heads * b)
    q = rearrange(q, "(b x1 h) n d -> (x1 h) (b n) d", h=num_heads, x1=X)
    k = rearrange(k, "(b x1 h) n d -> (x1 h) (b n) d", h=num_heads, x1=X)
    v = rearrange(v, "(b x1 h) n d -> (x1 h) (b n) d", h=num_heads, x1=X)
    out = xformers.ops.memory_efficient_attention(q, k, v)
    out = rearrange(out, "(x1 h) (b n) d -> (b x1) n (h d)", b=b, x1=X, h=num_heads)
    return out

# 模拟输入
B = 2
T = 16   # 时间步 or 视角数
H = 10   # sequence length
D = 96   # embedding dim
num_heads = 8
head_dim = D // num_heads

total_tokens = B * T * num_heads
q_all = torch.randn(total_tokens, H, head_dim)
k1 = torch.randn(total_tokens // 2, H, head_dim)  # only half: for q1
v1 = torch.randn_like(k1)

# split q_all = [q1, q2]
q1 = q_all[: total_tokens // 2]
q2 = q_all[total_tokens // 2 :]

# Case 1: q1 alone
out1 = xformers.ops.memory_efficient_attention(q1, k1, v1)
out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=num_heads)

# Case 2: q_all then取前半部分
out_all = efficient_attn(q_all, k1, v1, num_heads=num_heads, b=B)
out2 = out_all[:out_all.shape[0]//2]

# 比较结果是否一致
print("All close:", torch.allclose(out1, out2, atol=1e-5))
print("Max difference:", (out1 - out2).abs().max().item())