import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    n_embd: int
    n_heads: int
    dropout: float
    vocab_size: int
    max_length: int
    n_layer: int


'''
shape is like [B, Tq, n_embd], consider [Tq, n_embd] below:
q:                       k:                        v:
[q11, q12, q13, q14]      [k11, k12, k13, k14]      [v11, v12, v13, v14]
[q21, q22, q22, q24]      [k21, k22, k22, k24]      [v21, v22, v22, v24]
[pad, pad, pad, pad]      [pad, pad, pad, pad]      [pad, pad, pad, pad]

cos_similarity = qk^T
[c11, c12, c13(pad)]
[c21, c22, c33(pad)]
[pad, pad, pad]
'''
class Attention(nn.Module):

    def __init__(self, args: ModelArgs, casual = False):
        super().__init__()
        self.head_dim = args.n_embd // args.n_heads
        self.casual = casual

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_pad_mask: torch.Tensor,
    ):
        scale_factor = 1 / math.sqrt(q.size(-1))
        scaled_dot = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

        h, w = scaled_dot.size(-2), scaled_dot.size(-1)
        if self.casual:
            causal_mask = torch.ones(h, w, dtype=torch.bool, device=scaled_dot.device).triu(diagonal=1)
            scaled_dot = scaled_dot.masked_fill(causal_mask, float("-inf"))

        if k_pad_mask is not None:
            k_pad_mask = k_pad_mask.to(device=scaled_dot.device, dtype=torch.bool)
            scaled_dot = scaled_dot.masked_fill(k_pad_mask[:, None, :], float("-inf"))

        scores = torch.softmax(scaled_dot, dim=-1)
        return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, args: ModelArgs, causal=False):
        super().__init__()
        assert args.n_embd % args.n_heads == 0
        self.head_dim = args.n_embd // args.n_heads
        self.n_heads = args.n_heads
        self.attention = Attention(args, causal)
        self.wq = nn.ModuleList()
        self.wk = nn.ModuleList()
        self.wv = nn.ModuleList()
        for i in range(self.n_heads):
            self.wq.append(nn.Linear(args.n_embd, self.head_dim, bias=False))
            self.wk.append(nn.Linear(args.n_embd, self.head_dim, bias=False))
            self.wv.append(nn.Linear(args.n_embd, self.head_dim, bias=False))
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.n_embd, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_pad_mask: torch.Tensor,
    ):
        head_out = []
        for i in range(self.n_heads):
            q_i = self.wq[i](q)   # res: (B, Tq, head_dim)
            k_i = self.wk[i](k)
            v_i = self.wv[i](v)

            attention_i = self.attention.forward(
                q_i,
                k_i,
                v_i,
                k_pad_mask,
            )
            head_out.append(attention_i)
        
        head_concat = torch.cat(head_out, dim=-1)
        return self.wo(head_concat)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))
    

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_norm = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.n_embd, args.n_embd, args.dropout)

    def forward(self, x, k_pad_mask: torch.Tensor):
        h = self.attention_norm(x + self.attention.forward(x, x, x, k_pad_mask))
        out = self.fnn_norm(h + self.feed_forward.forward(h))
        return out


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__() 
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])

    def forward(self, x, k_pad_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, k_pad_mask)
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_norm_1 = LayerNorm(args.n_embd)
        self.mask_attention = MultiHeadAttention(args, causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args, causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.n_embd, args.n_embd, args.dropout)

    def forward(
        self,
        x,
        encoder_out,
        self_k_pad_mask: torch.Tensor = None,
        cross_k_pad_mask: torch.Tensor = None,
    ):
        h1 = self.attention_norm_1(
            x + self.mask_attention.forward(x, x, x, self_k_pad_mask)
        )
        h2 = self.attention_norm_2(
            h1 + self.attention.forward(h1, encoder_out, encoder_out, cross_k_pad_mask)
        )
        out = self.ffn_norm(
            h2 + self.feed_forward.forward(h2)
        )
        return out


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__() 
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(
        self,
        x,
        enc_out,
        self_k_pad_mask: torch.Tensor,
        cross_k_pad_mask: torch.Tensor,
    ):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(
                x,
                enc_out,
                self_k_pad_mask,
                cross_k_pad_mask,
            )
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(args.max_length, args.n_embd)
        position = torch.arange(0, args.max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_length is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            wpe=PositionalEncoding(args),
            drop=nn.Dropout(args.dropout),
            encoder=Encoder(args),
            decoder=Decoder(args),
        ))
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self,
        encoder_idx: torch.Tensor,
        encoder_k_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        _, src_t = encoder_idx.size()
        assert src_t <= self.args.max_length, f"encoder length is {src_t}, max is {self.args.max_length}"

        enc_tok_emb = self.transformer.wte(encoder_idx)
        enc_pos_emb = self.transformer.wpe(enc_tok_emb)
        enc_x = self.transformer.drop(enc_pos_emb)
        return self.transformer.encoder(enc_x, encoder_k_pad_mask)

    def forward(
        self,
        encoder_idx,
        decoder_idx,
        targets=None,
        debug=False,
        encoder_k_pad_mask: torch.Tensor=None,
        decoder_k_pad_mask: torch.Tensor=None,
    ):
        _, tgt_t = decoder_idx.size()
        assert tgt_t <= self.args.max_length, f"decoder length is {tgt_t}, max is {self.args.max_length}"

        if debug:
            print("encoder_idx", encoder_idx.size())
            print("decoder_idx", decoder_idx.size())

        enc_out = self.encode(encoder_idx, encoder_k_pad_mask)
        if debug:
            print("enc_out:", enc_out.size())

        dec_tok_emb = self.transformer.wte(decoder_idx)
        dec_pos_emb = self.transformer.wpe(dec_tok_emb)
        dec_x = self.transformer.drop(dec_pos_emb)
        dec_x = self.transformer.decoder(
            dec_x,
            enc_out,
            decoder_k_pad_mask,
            encoder_k_pad_mask,
        )
        if debug:
            print("dec_x:", dec_x.size())

        if targets is not None:
            logits = self.lm_head(dec_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(dec_x[:, [-1], :])
            loss = None

        return logits, loss
