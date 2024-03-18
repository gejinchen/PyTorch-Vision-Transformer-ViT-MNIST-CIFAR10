import torch
import torch.nn as nn

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        x = x.reshape([x.shape[0], x.shape[1], -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.transpose(1, 2)  # B E S -> B S E 
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence
        x = x + self.pos_embedding  # Adding positional embedding
        return x


# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, n_attention_heads):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.n_attention_heads = n_attention_heads
#         self.head_embed_dim = embed_dim // n_attention_heads

#         self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
#         self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
#         self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)

#     def forward(self, x):
#         B, S, E = x.shape

#         xq = self.queries(x).reshape(B, S, self.n_attention_heads, self.head_embed_dim)  # B, S, E -> B, S, H, HE
#         xq = xq.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE
#         xk = self.keys(x).reshape(B, S, self.n_attention_heads, self.head_embed_dim)  # B, S, E -> B, S, H, HE
#         xk = xk.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE
#         xv = self.values(x).reshape(B, S, self.n_attention_heads, self.head_embed_dim)  # B, S, E -> B, S, H, HE
#         xv = xv.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE

#         xq = xq.reshape([-1, S, self.head_embed_dim])  # B, H, S, HE -> (BH), S, HE
#         xk = xk.reshape([-1, S, self.head_embed_dim])  # B, H, S, HE -> (BH), S, HE
#         xv = xv.reshape([-1, S, self.head_embed_dim])  # B, H, S, HE -> (BH), S, HE

#         xk = xk.transpose(1, 2)  # (BH), S, HE -> (BH), HE, S
#         x_attention = xq.bmm(xk)  # (BH), S, HE  .  (BH), HE, S -> (BH), S, S    ========================= should do / sqrt(dk)?
#         x_attention = torch.softmax(x_attention, dim=-1)

#         x = x_attention.bmm(xv)  # (BH), S, S . (BH), S, HE -> (BH), S, HE
#         x = x.reshape([-1, self.n_attention_heads, S, self.head_embed_dim])  # (BH), S, HE -> B, H, S, HE
#         x = x.transpose(1, 2)  # B, H, S, HE -> B, S, H, HE
#         x = x.reshape(B, S, E)  # B, S, H, HE -> B, S, E
#         return x
    

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.queries = nn.Linear(self.embed_dim, self.embed_dim)
        self.keys = nn.Linear(self.embed_dim, self.embed_dim)
        self.values = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        B, S, E = x.shape

        # linear layers
        xq = self.queries(x) # B, S, E -> B, S, E
        xk = self.keys(x) # B, S, E -> B, S, E
        xv = self.values(x) # B, S, E -> B, S, E

        # split heads
        xq = xq.view(B, S, self.num_heads, self.head_dim)  # B, S, E -> B, S, H, HE
        xk = xk.view(B, S, self.num_heads, self.head_dim)  # B, S, E -> B, S, H, HE
        xv = xv.view(B, S, self.num_heads, self.head_dim)  # B, S, E -> B, S, H, HE

        # reshape
        xq = xq.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE
        xk = xk.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE
        xv = xv.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE

        # self attention
        xk = xk.transpose(-1, -2)  # B, H, S, HE -> B, H, HE, S
        x_attn = torch.matmul(xq, xk)  # B, H, S, HE  *  B, H, HE, S -> B, H, S, S
        # # scale
        # x_attn /= float(self.head_dim) ** 0.5
        x_attn = torch.softmax(x_attn, dim=-1)

        # apply attention
        x = torch.matmul(x_attn, xv)  # B, H, S, S * B, H, S, HE -> B, H, S, HE

        # concatenate heads
        x = x.transpose(1, 2)  # B, H, S, HE -> B, S, H, HE
        x = x.reshape(B, S, E)  # B, S, H, HE -> B, S, E
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul):
        super().__init__()
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x)) # Skip connections
        x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))  # Skip connections
        return x


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # Newer architectures skip fc1 and activations and directly apply fc2.
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size)
        self.encoder = nn.Sequential(*[Encoder(embed_dim, n_attention_heads, forward_mul) for _ in range(n_layers)], nn.LayerNorm(embed_dim))
        self.norm = nn.LayerNorm(embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x
