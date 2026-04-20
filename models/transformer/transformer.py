import torch
import math
from torch import nn
from dataclasses import asdict, dataclass
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

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


class DataPipeline:
    def __init__(self):
        self._worker_state = threading.local()

    def _tokenize_translation_batch(self, tokenizer, src_texts, tgt_texts, max_length):
        src_encoded = tokenizer(
            src_texts,
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        tgt_encoded = tokenizer(
            tgt_texts,
            max_length=max_length - 1,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        return src_encoded.input_ids, tgt_encoded.input_ids

    def _get_worker_tokenizer(self, tokenizer_name, tokenizer_use_fast):
        worker_tokenizer = getattr(self._worker_state, "tokenizer", None)
        cached_name = getattr(self._worker_state, "tokenizer_name", None)
        cached_use_fast = getattr(self._worker_state, "tokenizer_use_fast", None)
        if (
            worker_tokenizer is None
            or cached_name != tokenizer_name
            or cached_use_fast != tokenizer_use_fast
        ):
            worker_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=tokenizer_use_fast,
                local_files_only=True,
            )
            self._worker_state.tokenizer = worker_tokenizer
            self._worker_state.tokenizer_name = tokenizer_name
            self._worker_state.tokenizer_use_fast = tokenizer_use_fast
        return worker_tokenizer

    def _tokenize_sub_batch(
        self,
        tokenizer_name,
        tokenizer_use_fast,
        src_batch,
        tgt_batch,
        max_length,
    ):
        worker_tokenizer = self._get_worker_tokenizer(tokenizer_name, tokenizer_use_fast)
        return self._tokenize_translation_batch(
            worker_tokenizer,
            src_batch,
            tgt_batch,
            max_length,
        )

    def pad_token_id(self, tokenizer):
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        return pad_token_id

    def bos_token_id(self, tokenizer):
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.cls_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.sep_token_id
        if bos_token_id is None:
            bos_token_id = 0
        return bos_token_id

    def eos_token_id(self, tokenizer):
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.sep_token_id
        return eos_token_id

    def build_translation_seq2seq_dataset(
        self,
        tokenizer,
        chunk_data,
        max_length,
        tokenize_batch_size=20000,
        tokenize_workers=16,
        tokenize_log_interval=10,
        progress_prefix="",
    ):
        src_texts, tgt_texts = chunk_data
        if len(src_texts) != len(tgt_texts):
            raise ValueError("Source and target sample counts do not match.")

        tokenize_batch_size = max(1, tokenize_batch_size)
        tokenize_log_interval = max(1, tokenize_log_interval)
        progress_prefix = f"{progress_prefix} " if progress_prefix else ""
        max_length = max(4, max_length)
        sub_batches = [
            (
                src_texts[i : i + tokenize_batch_size],
                tgt_texts[i : i + tokenize_batch_size],
            )
            for i in range(0, len(src_texts), tokenize_batch_size)
        ]
        if not sub_batches:
            raise ValueError("No translation samples found in this chunk.")
        total_tasks = len(sub_batches)
        tokenizer_name = tokenizer.name_or_path
        tokenizer_use_fast = getattr(tokenizer, "is_fast", True)

        def maybe_print_tokenize_progress(done_tasks):
            if (
                done_tasks == 1
                or done_tasks == total_tasks
                or done_tasks % tokenize_log_interval == 0
            ):
                progress = (done_tasks / total_tasks) * 100
                print(
                    f"{progress_prefix} chunk loaded progress {done_tasks}/{total_tasks} ({progress:.1f}%)",
                    flush=True,
                )

        max_workers = min(max(1, tokenize_workers), len(sub_batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._tokenize_sub_batch,
                    tokenizer_name,
                    tokenizer_use_fast,
                    src_batch,
                    tgt_batch,
                    max_length,
                ): idx
                for idx, (src_batch, tgt_batch) in enumerate(sub_batches)
            }
            src_parts = [None] * len(sub_batches)
            tgt_parts = [None] * len(sub_batches)
            done_tasks = 0
            for future in as_completed(futures):
                idx = futures[future]
                src_ids, tgt_ids = future.result()
                src_parts[idx] = src_ids
                tgt_parts[idx] = tgt_ids
                done_tasks += 1
                maybe_print_tokenize_progress(done_tasks)

        dataset = []
        for src_batch_ids, tgt_batch_ids in zip(src_parts, tgt_parts):
            dataset.extend(zip(src_batch_ids, tgt_batch_ids))
        return dataset

    def collate_translation_batch(self, batch, tokenizer):
        pad_token_id = self.pad_token_id(tokenizer)
        bos_token_id = self.bos_token_id(tokenizer)
        eos_token_id = self.eos_token_id(tokenizer)

        batch_encoder_inputs = pad_sequence(
            [torch.tensor(src_ids, dtype=torch.long) for src_ids, _ in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        batch_decoder_inputs = pad_sequence(
            [torch.tensor([bos_token_id, *tgt_ids], dtype=torch.long) for _, tgt_ids in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        batch_targets = pad_sequence(
            [torch.tensor([*tgt_ids, eos_token_id], dtype=torch.long) for _, tgt_ids in batch],
            batch_first=True,
            padding_value=-100,
        )
        return batch_encoder_inputs, batch_decoder_inputs, batch_targets

    def init_read_data_chunk_iter(
        self,
        dataset_name,
        dataset_config,
        dataset_split,
        src_lang,
        tgt_lang,
        chunk_lines,
        data_offset=0,
        max_samples=None,
        shuffle_dataset=False,
        shuffle_seed=42,
    ):
        if load_dataset is None:
            raise ImportError("datasets is not installed. Please run: pip install datasets")

        chunk_lines = max(1, chunk_lines)
        data_offset = max(0, data_offset)
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        if shuffle_dataset:
            dataset = dataset.shuffle(seed=shuffle_seed)
        emitted = 0
        src_chunk = []
        tgt_chunk = []
        for row_idx, item in enumerate(dataset):
            if row_idx < data_offset:
                continue
            pair = item.get("translation")
            if not isinstance(pair, dict):
                continue
            src_text = pair.get(src_lang)
            tgt_text = pair.get(tgt_lang)
            if src_text is None or tgt_text is None:
                continue
            src_text = str(src_text).strip()
            tgt_text = str(tgt_text).strip()
            if not src_text or not tgt_text:
                continue
            src_chunk.append(src_text)
            tgt_chunk.append(tgt_text)
            emitted += 1
            if len(src_chunk) >= chunk_lines:
                yield src_chunk, tgt_chunk
                src_chunk = []
                tgt_chunk = []
            if max_samples is not None and emitted >= max_samples:
                break

        if src_chunk:
            yield src_chunk, tgt_chunk

class TransformerCLI:
    def __init__(self):
        self.data_pipeline = DataPipeline()
        self.resume_ckpt = None
        self.infer_ckpt = None
        self.runtime_device = None
        self.tokenizer = None
        self.pad_token_id = None
        self.model_args = None
        self.parse_cli_args()
        self.init_runtime_state()

    def apply_checkpoint_tokenizer_name(self, ckpt, checkpoint_path):
        checkpoint_tokenizer_name = ckpt.get("tokenizer_name")
        if checkpoint_tokenizer_name is None:
            return
        if checkpoint_tokenizer_name != self.tokenizer_name:
            print(
                f"using tokenizer_name from checkpoint {checkpoint_path}: {checkpoint_tokenizer_name} "
                f"(overriding cli --tokenizer_name={self.tokenizer_name})",
                flush=True,
            )
        self.tokenizer_name = checkpoint_tokenizer_name

    def init_runtime_state(self):
        self.runtime_device = torch.device(self.device)
        if self.mode == "train" and self.resume_from_checkpoint is not None:
            self.resume_ckpt = torch.load(self.resume_from_checkpoint, map_location=self.runtime_device)
            self.apply_checkpoint_tokenizer_name(self.resume_ckpt, self.resume_from_checkpoint)
            print(f"resuming from checkpoint: {self.resume_from_checkpoint}", flush=True)
        if self.mode == "infer":
            self.infer_ckpt = torch.load(self.checkpoint, map_location=self.runtime_device)
            self.apply_checkpoint_tokenizer_name(self.infer_ckpt, self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        self.pad_token_id = self.data_pipeline.pad_token_id(self.tokenizer)
        if self.mode == "train":
            self.model_args = self.init_model_args()

    def init_model_args(self):
        if self.resume_ckpt is not None and "model_args" in self.resume_ckpt:
            model_args = ModelArgs(**self.resume_ckpt["model_args"])
        else:
            model_args = ModelArgs(
                n_embd=512,
                n_heads=8,
                dropout=0.1,
                vocab_size=self.tokenizer.vocab_size,
                max_length=self.max_length,
                n_layer=6,
            )
        if model_args.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"Tokenizer vocab_size ({self.tokenizer.vocab_size}) does not match model vocab_size ({model_args.vocab_size})."
            )
        if model_args.max_length != self.max_length:
            print(
                f"using max_length from model/checkpoint: {model_args.max_length} "
                f"(ignoring cli --max_length={self.max_length})",
                flush=True,
            )
        return model_args

    def save_checkpoint(
        self,
        path,
        model,
        model_args,
    ):
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_args": asdict(model_args),
            "tokenizer_name": self.tokenizer_name,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ckpt, path)

    def load_checkpoint(self, path, device, ckpt=None):
        if ckpt is None:
            ckpt = torch.load(path, map_location=device)
        model_args = ModelArgs(**ckpt["model_args"])
        model = Transformer(model_args).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    @torch.no_grad()
    def generate_greedy(
        self,
        model,
        encoder_idx,
        decoder_idx,
        max_new_tokens,
        eos_token_id=None,
        encoder_k_pad_mask=None,
    ):
        model.eval()
        for _ in range(max_new_tokens):
            encoder_cond = encoder_idx[:, -model.args.max_length:]
            decoder_cond = decoder_idx[:, -model.args.max_length:]
            encoder_mask_cond = None
            if encoder_k_pad_mask is not None:
                encoder_mask_cond = encoder_k_pad_mask[:, -model.args.max_length:]
            logits, _ = model(
                encoder_cond,
                decoder_cond,
                encoder_k_pad_mask=encoder_mask_cond,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            decoder_idx = torch.cat([decoder_idx, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
        return decoder_idx

    @torch.no_grad()
    def print_encoder_out_diagnostics(self, model, encoder_idx, encoder_k_pad_mask):
        encoder_out = model.encode(encoder_idx, encoder_k_pad_mask)
        valid_token_count = int((~encoder_k_pad_mask[0]).sum().item())
        print("encoder_out shape:", tuple(encoder_out.shape))
        print("encoder_out valid tokens:", valid_token_count)
        if valid_token_count > 0:
            valid_encoder_out = encoder_out[0, :valid_token_count].detach().cpu()
            print("encoder_out stats:", {
                "mean": float(valid_encoder_out.mean().item()),
                "std": float(valid_encoder_out.std().item()),
                "min": float(valid_encoder_out.min().item()),
                "max": float(valid_encoder_out.max().item()),
            })
            print(
                "encoder_out per_token_std:",
                valid_encoder_out.std(dim=-1, unbiased=False),
            )
            if valid_token_count > 1:
                print(
                    "encoder_out adjacent cosine_similarity:",
                    F.cosine_similarity(
                        valid_encoder_out[:-1],
                        valid_encoder_out[1:],
                        dim=-1,
                    ),
                )
                print(
                    "encoder_out adjacent l2_diff:",
                    torch.norm(
                        valid_encoder_out[1:] - valid_encoder_out[:-1],
                        dim=-1,
                    ),
                )
            else:
                print("encoder_out adjacent comparisons: skipped, only one valid token")
            print("encoder_out:", valid_encoder_out)
        else:
            print("encoder_out stats: skipped, no valid encoder tokens")

    def parse_cli_args(self):
        parser = argparse.ArgumentParser(description="Train and run a tiny Transformer demo.")
        parser.add_argument("--mode", choices=["train", "infer"], default="train")
        parser.add_argument("--checkpoint", type=str, default="transformer.pt")
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="Resume training from a saved checkpoint path",
        )
        parser.add_argument("--tokenizer_name", type=str, default="bert-base-multilingual-cased")
        parser.add_argument("--hf_dataset_name", type=str, default="wmt/wmt18")
        parser.add_argument("--hf_dataset_config", type=str, default="zh-en")
        parser.add_argument("--hf_dataset_split", type=str, default="train")
        parser.add_argument("--src_lang", type=str, default="zh", help="Source language code")
        parser.add_argument("--tgt_lang", type=str, default="en", help="Target language code")
        parser.add_argument("--text", type=str, default="I love learning large language models.")
        parser.add_argument("--epochs", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument(
            "--optimizer",
            choices=["adam", "adamw"],
            default="adam",
            help="Training optimizer; default is Adam to avoid decoupled weight decay shrinking LayerNorm params",
        )
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--log_interval", type=int, default=100, help="Print train progress every N batches")
        parser.add_argument(
            "--max_samples",
            type=int,
            default=0,
            help="Maximum number of training samples to read; 0 means no limit",
        )
        parser.add_argument(
            "--chunk_lines",
            type=int,
            default=200000,
            help="Number of translation pairs per chunk",
        )
        parser.add_argument(
            "--data_offset",
            type=int,
            default=0,
            help="Number of dataset rows to skip before reading training data",
        )
        parser.add_argument(
            "--shuffle_dataset",
            action="store_true",
            help="Globally shuffle the HF dataset before splitting it into chunks",
        )
        parser.add_argument(
            "--shuffle_seed",
            type=int,
            default=42,
            help="Base random seed for global dataset shuffle; each epoch adds epoch - 1",
        )
        parser.add_argument(
            "--tokenize_batch_size",
            type=int,
            default=2000,
            help="Tokenization mini-batch size inside each chunk",
        )
        parser.add_argument(
            "--tokenize_workers",
            type=int,
            default=16,
            help="Number of parallel tokenization threads per chunk (1 disables parallelism)",
        )
        parser.add_argument(
            "--tokenize_log_interval",
            type=int,
            default=10,
            help="Print tokenization progress every N tokenization tasks",
        )
        parser.add_argument("--max_length", type=int, default=128)
        parser.add_argument("--max_new_tokens", type=int, default=20)
        parser.add_argument(
            "--print_encoder_out",
            action="store_true",
            help="Print encoder output once before greedy decoding during inference",
        )
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        cli_args = parser.parse_args()
        self.cli_args = cli_args
        for key, value in vars(cli_args).items():
            setattr(self, key, value)

    def build_optimizer(self, model):
        if self.optimizer == "adam":
            print(f"using optimizer: Adam lr={self.lr}", flush=True)
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        # AdamW applies decoupled weight decay to all parameters here, including
        # LayerNorm weights, which can undesirably shrink LayerNorm scales. 
        # And will cause encoder collapse
        print(f"using optimizer: AdamW lr={self.lr}", flush=True)
        return torch.optim.AdamW(model.parameters(), lr=self.lr)

    def run_train(self):
        device = self.runtime_device
        tokenizer = self.tokenizer
        pad_token_id = self.pad_token_id
        model_args = self.model_args
        train_max_length = model_args.max_length
        model = Transformer(model_args).to(device)
        if self.resume_ckpt is not None:
            model.load_state_dict(self.resume_ckpt["model_state_dict"])
            print("loaded model state from checkpoint", flush=True)
        chunk_lines = max(1, self.chunk_lines)
        data_offset = max(0, self.data_offset)
        max_samples = None if self.max_samples == 0 else max(1, self.max_samples)
        tokenize_batch_size = max(1, self.tokenize_batch_size)
        tokenize_workers = max(1, self.tokenize_workers)
        tokenize_log_interval = max(1, self.tokenize_log_interval)
        optimizer = self.build_optimizer(model)
        model.train()
        def collate_translation_batch(batch):
            return self.data_pipeline.collate_translation_batch(batch, tokenizer)

        log_interval = max(1, self.log_interval)
        global_step = 0
        try:
            for epoch in range(1, self.epochs + 1):
                epoch_loss_sum = 0.0
                epoch_batches = 0
                epoch_chunks = 0
                print(
                    f"epoch {epoch}/{self.epochs} loading HF dataset "
                    f"{self.hf_dataset_name} ({self.hf_dataset_config}) "
                    f"split={self.hf_dataset_split} offset={data_offset} "
                    f"{self.src_lang}->{self.tgt_lang} "
                    f"shuffle={self.shuffle_dataset} seed={self.shuffle_seed + epoch - 1}",
                    flush=True,
                )
                data_chunk_iter = self.data_pipeline.init_read_data_chunk_iter(
                    self.hf_dataset_name,
                    self.hf_dataset_config,
                    self.hf_dataset_split,
                    self.src_lang,
                    self.tgt_lang,
                    chunk_lines,
                    data_offset=data_offset,
                    max_samples=max_samples,
                    shuffle_dataset=self.shuffle_dataset,
                    shuffle_seed=self.shuffle_seed + epoch - 1,
                )

                def next_chunk_dataset(chunk_idx):
                    chunk_data = next(data_chunk_iter, None)
                    if chunk_data is None:
                        return None
                    dataset = self.data_pipeline.build_translation_seq2seq_dataset(
                        tokenizer,
                        chunk_data,
                        train_max_length,
                        tokenize_batch_size=tokenize_batch_size,
                        tokenize_workers=tokenize_workers,
                        tokenize_log_interval=tokenize_log_interval,
                        progress_prefix=f"epoch {epoch}/{self.epochs} chunk {chunk_idx}",
                    )
                    return dataset

                chunk_idx = 1
                while (dataset := next_chunk_dataset(chunk_idx)) is not None:
                    epoch_chunks = chunk_idx
                    dataloader = DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        collate_fn=collate_translation_batch,
                    )
                    total_batches = len(dataloader)
                    print(
                        f"epoch {epoch}/{self.epochs} "
                        f"chunk {chunk_idx} loaded successfully ",
                        flush=True,
                    )

                    for batch_idx, (batch_encoder_inputs, batch_decoder_inputs, batch_targets) in enumerate(dataloader, start=1):
                        batch_encoder_inputs = batch_encoder_inputs.to(device)
                        batch_decoder_inputs = batch_decoder_inputs.to(device)
                        batch_targets = batch_targets.to(device)
                        batch_encoder_k_pad_mask = batch_encoder_inputs.eq(pad_token_id)
                        batch_decoder_k_pad_mask = batch_decoder_inputs.eq(pad_token_id)
                        logits, next_token_loss = model(
                            batch_encoder_inputs,
                            batch_decoder_inputs,
                            batch_targets,
                            encoder_k_pad_mask=batch_encoder_k_pad_mask,
                            decoder_k_pad_mask=batch_decoder_k_pad_mask,
                        )
                        first_token_loss = F.cross_entropy(
                            logits[:, 0, :].detach(),
                            batch_targets[:, 0],
                            ignore_index=-100,
                        )
                        loss = next_token_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        global_step += 1
                        epoch_batches += 1
                        epoch_loss_sum += loss.item()
                        if global_step == 1 or global_step % log_interval == 0 or batch_idx == total_batches:
                            epoch_avg_so_far = epoch_loss_sum / epoch_batches
                            chunk_progress = (batch_idx / max(total_batches, 1)) * 100
                            print(
                                f"epoch {epoch}/{self.epochs} "
                                f"chunk {chunk_idx} batch {batch_idx}/{total_batches} ({chunk_progress:.1f}%) "
                                f"step {global_step} "
                                f"loss={loss.item():.4f} "
                                f"next_loss={next_token_loss.item():.4f} "
                                f"first_loss={first_token_loss.item():.4f} "
                                f"avg_loss={epoch_avg_so_far:.4f}",
                                flush=True,
                            )
                    chunk_idx = chunk_idx + 1

                if epoch_batches == 0:
                    raise ValueError("No training samples were produced from the HF dataset.")
                epoch_avg_loss = epoch_loss_sum / epoch_batches
                print(
                    f"epoch {epoch}/{self.epochs} avg_loss={epoch_avg_loss:.4f} "
                    f"chunks={epoch_chunks}",
                    flush=True,
                )
        except KeyboardInterrupt:
            print(
                f"\nTraining interrupted at step {global_step}. Saving checkpoint...",
                flush=True,
            )
            return
        except Exception as e:
            print(f"training failed: {e}", flush=True)
            raise
        finally:
            self.save_checkpoint(
                self.checkpoint,
                model,
                model_args,
            )
            print(f"checkpoint saved: {self.checkpoint}", flush=True)

    def run_infer(self):
        device = self.runtime_device
        model = self.load_checkpoint(self.checkpoint, device, ckpt=self.infer_ckpt)
        tokenizer = self.tokenizer
        pad_token_id = self.pad_token_id
        encoder_idx = tokenizer(
            self.text,
            return_tensors="pt",
            max_length=model.args.max_length,
            truncation=True,
        ).input_ids.to(device)
        encoder_k_pad_mask = encoder_idx.eq(pad_token_id)
        if self.print_encoder_out:
            self.print_encoder_out_diagnostics(model, encoder_idx, encoder_k_pad_mask)
        bos_token_id = self.data_pipeline.bos_token_id(tokenizer)
        eos_token_id = self.data_pipeline.eos_token_id(tokenizer)
        decoder_start = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)
        generated_decoder = self.generate_greedy(
            model,
            encoder_idx,
            decoder_start,
            self.max_new_tokens,
            eos_token_id=eos_token_id,
            encoder_k_pad_mask=encoder_k_pad_mask,
        )
        print(tokenizer.decode(generated_decoder[0, 1:], skip_special_tokens=True))

    def main(self):
        if self.mode == "train":
            self.run_train()
        else:
            self.run_infer()


if __name__ == "__main__":
    print("start")
    TransformerCLI().main()
