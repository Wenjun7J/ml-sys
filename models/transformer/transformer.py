import torch
import math
from torch import nn
from dataclasses import asdict, dataclass
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
    block_size: int
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
            q_i = self.wq[i](q)   # (B, Tq, head_dim)
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
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
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
        assert args.block_size is not None
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

    def forward(
        self,
        encoder_idx,
        decoder_idx,
        targets=None,
        debug=False,
        encoder_k_pad_mask: torch.Tensor=None,
        decoder_k_pad_mask: torch.Tensor=None,
    ):
        _, src_t = encoder_idx.size()
        _, tgt_t = decoder_idx.size()
        assert src_t <= self.args.block_size, f"encoder length is {src_t}, max is {self.args.block_size}"
        assert tgt_t <= self.args.block_size, f"decoder length is {tgt_t}, max is {self.args.block_size}"

        if debug:
            print("encoder_idx", encoder_idx.size())
            print("decoder_idx", decoder_idx.size())

        enc_tok_emb = self.transformer.wte(encoder_idx)
        enc_pos_emb = self.transformer.wpe(enc_tok_emb)
        enc_x = self.transformer.drop(enc_pos_emb)
        enc_out = self.transformer.encoder(enc_x, encoder_k_pad_mask)
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


def _tokenize_translation_batch(tokenizer, src_texts, tgt_texts, max_length):
    src_encoded = tokenizer(
        src_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    tgt_encoded = tokenizer(
        tgt_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return src_encoded.input_ids, tgt_encoded.input_ids


def build_translation_seq2seq_dataset(
    tokenizer,
    src_texts,
    tgt_texts,
    block_size,
    tokenize_batch_size=20000,
    tokenize_workers=16,
    tokenize_log_interval=10,
    progress_prefix="",
):
    if len(src_texts) != len(tgt_texts):
        raise ValueError("Source and target sample counts do not match.")

    tokenize_batch_size = max(1, tokenize_batch_size)
    tokenize_log_interval = max(1, tokenize_log_interval)
    progress_prefix = f"{progress_prefix} " if progress_prefix else ""
    max_length = max(4, block_size)
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

    def maybe_print_tokenize_progress(done_tasks):
        if (
            done_tasks == 1
            or done_tasks == total_tasks
            or done_tasks % tokenize_log_interval == 0
        ):
            progress = (done_tasks / total_tasks) * 100
            print(
                f"{progress_prefix}tokenize task {done_tasks}/{total_tasks} ({progress:.1f}%)",
                flush=True,
            )

    if tokenize_workers <= 1 or len(sub_batches) == 1:
        src_parts = []
        tgt_parts = []
        for task_idx, (src_batch, tgt_batch) in enumerate(sub_batches, start=1):
            src_ids, tgt_ids = _tokenize_translation_batch(
                tokenizer, src_batch, tgt_batch, max_length
            )
            src_parts.append(src_ids)
            tgt_parts.append(tgt_ids)
            maybe_print_tokenize_progress(task_idx)
    else:
        max_workers = min(max(1, tokenize_workers), len(sub_batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _tokenize_translation_batch, tokenizer, src_batch, tgt_batch, max_length
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

    encoder_tensor = torch.cat(src_parts, dim=0)
    target_tokens = torch.cat(tgt_parts, dim=0)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.cls_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.sep_token_id
    if bos_token_id is None:
        bos_token_id = pad_token_id

    decoder_tensor = torch.full_like(target_tokens, pad_token_id)
    decoder_tensor[:, 0] = bos_token_id
    decoder_tensor[:, 1:] = target_tokens[:, :-1]
    labels = target_tokens.clone()
    labels = labels.masked_fill(labels == pad_token_id, -100)
    return TensorDataset(encoder_tensor, decoder_tensor, labels)


def iter_hf_translation_chunks(
    dataset_name,
    dataset_config,
    dataset_split,
    src_lang,
    tgt_lang,
    chunk_lines,
    max_samples=None,
):
    if load_dataset is None:
        raise ImportError("datasets is not installed. Please run: pip install datasets")

    chunk_lines = max(1, chunk_lines)
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    emitted = 0
    src_chunk = []
    tgt_chunk = []
    for item in dataset:
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


def save_checkpoint(
    path,
    model,
    optimizer,
    model_args,
    tokenizer_name,
    global_step=None,
    epoch=None,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_args": asdict(model_args),
        "tokenizer_name": tokenizer_name,
    }
    if global_step is not None:
        ckpt["global_step"] = int(global_step)
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model_args = ModelArgs(**ckpt["model_args"])
    tokenizer_name = ckpt.get("tokenizer_name", "bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = Transformer(model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_greedy(
    model,
    encoder_idx,
    decoder_idx,
    max_new_tokens,
    eos_token_id=None,
    encoder_k_pad_mask=None,
):
    model.eval()
    for _ in range(max_new_tokens):
        encoder_cond = encoder_idx[:, -model.args.block_size:]
        decoder_cond = decoder_idx[:, -model.args.block_size:]
        encoder_mask_cond = None
        if encoder_k_pad_mask is not None:
            encoder_mask_cond = encoder_k_pad_mask[:, -model.args.block_size:]
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


def parse_cli_args():
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
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_train(cli_args):
    device = torch.device(cli_args.device)
    resume_ckpt = None
    tokenizer_name = cli_args.tokenizer_name
    if cli_args.resume_from_checkpoint is not None:
        resume_ckpt = torch.load(cli_args.resume_from_checkpoint, map_location=device)
        tokenizer_name = resume_ckpt.get("tokenizer_name", tokenizer_name)
        print(f"resuming from checkpoint: {cli_args.resume_from_checkpoint}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if resume_ckpt is not None and "model_args" in resume_ckpt:
        model_args = ModelArgs(**resume_ckpt["model_args"])
    else:
        model_args = ModelArgs(
            n_embd=512,
            n_heads=8,
            dropout=0.1,
            vocab_size=tokenizer.vocab_size,
            block_size=cli_args.block_size,
            n_layer=6,
        )
    if model_args.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match model vocab_size ({model_args.vocab_size})."
        )
    if model_args.block_size != cli_args.block_size:
        print(
            f"using block_size from model/checkpoint: {model_args.block_size} "
            f"(ignoring cli --block_size={cli_args.block_size})",
            flush=True,
        )
    train_block_size = model_args.block_size
    model = Transformer(model_args).to(device)
    chunk_lines = max(1, cli_args.chunk_lines)
    max_samples = None if cli_args.max_samples == 0 else max(1, cli_args.max_samples)
    tokenize_batch_size = max(1, cli_args.tokenize_batch_size)
    tokenize_workers = max(1, cli_args.tokenize_workers)
    tokenize_log_interval = max(1, cli_args.tokenize_log_interval)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cli_args.lr)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print("loaded model/optimizer state from checkpoint", flush=True)
    model.train()

    log_interval = max(1, cli_args.log_interval)
    global_step = int(resume_ckpt.get("global_step", 0)) if resume_ckpt is not None else 0
    last_epoch = int(resume_ckpt.get("epoch", 0)) if resume_ckpt is not None else 0
    try:
        for epoch in range(1, cli_args.epochs + 1):
            last_epoch = epoch
            epoch_loss_sum = 0.0
            epoch_batches = 0
            epoch_samples = 0
            epoch_chunks = 0
            print(
                f"epoch {epoch}/{cli_args.epochs} loading HF dataset "
                f"{cli_args.hf_dataset_name} ({cli_args.hf_dataset_config}) "
                f"split={cli_args.hf_dataset_split} {cli_args.src_lang}->{cli_args.tgt_lang}",
                flush=True,
            )
            chunk_iter = iter_hf_translation_chunks(
                cli_args.hf_dataset_name,
                cli_args.hf_dataset_config,
                cli_args.hf_dataset_split,
                cli_args.src_lang,
                cli_args.tgt_lang,
                chunk_lines,
                max_samples=max_samples,
            )
            first_chunk_data = next(chunk_iter, None)
            if first_chunk_data is None:
                raise ValueError("No translation samples found in the HF dataset.")

            def submit_chunk_dataset(prefetch_executor, chunk_data, chunk_idx):
                src_texts, tgt_texts = chunk_data
                return prefetch_executor.submit(
                    build_translation_seq2seq_dataset,
                    tokenizer,
                    src_texts,
                    tgt_texts,
                    train_block_size,
                    tokenize_batch_size=tokenize_batch_size,
                    tokenize_workers=tokenize_workers,
                    tokenize_log_interval=tokenize_log_interval,
                    progress_prefix=f"epoch {epoch}/{cli_args.epochs} chunk {chunk_idx}",
                )

            def chunk_size(chunk_data):
                src_texts, _ = chunk_data
                return len(src_texts)

            with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                current_chunk_idx = 1
                current_chunk_data = first_chunk_data
                current_dataset_future = submit_chunk_dataset(
                    prefetch_executor, current_chunk_data, current_chunk_idx
                )

                while current_dataset_future is not None:
                    dataset = current_dataset_future.result()

                    next_chunk_data = next(chunk_iter, None)
                    if next_chunk_data is not None:
                        next_chunk_idx = current_chunk_idx + 1
                        next_dataset_future = submit_chunk_dataset(
                            prefetch_executor, next_chunk_data, next_chunk_idx
                        )
                    else:
                        next_dataset_future = None

                    epoch_chunks = current_chunk_idx
                    epoch_samples += chunk_size(current_chunk_data)
                    dataloader = DataLoader(dataset, batch_size=cli_args.batch_size, shuffle=True)
                    total_batches = len(dataloader)
                    print(
                        f"epoch {epoch}/{cli_args.epochs} "
                        f"chunk {current_chunk_idx} loaded {chunk_size(current_chunk_data)} samples "
                        f"(epoch_samples={epoch_samples})",
                        flush=True,
                    )

                    for batch_idx, (batch_encoder_inputs, batch_decoder_inputs, batch_targets) in enumerate(dataloader, start=1):
                        batch_encoder_inputs = batch_encoder_inputs.to(device)
                        batch_decoder_inputs = batch_decoder_inputs.to(device)
                        batch_targets = batch_targets.to(device)
                        batch_encoder_k_pad_mask = batch_encoder_inputs.eq(pad_token_id)
                        batch_decoder_k_pad_mask = batch_decoder_inputs.eq(pad_token_id)
                        _, loss = model(
                            batch_encoder_inputs,
                            batch_decoder_inputs,
                            batch_targets,
                            encoder_k_pad_mask=batch_encoder_k_pad_mask,
                            decoder_k_pad_mask=batch_decoder_k_pad_mask,
                        )
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
                                f"epoch {epoch}/{cli_args.epochs} "
                                f"chunk {current_chunk_idx} batch {batch_idx}/{total_batches} ({chunk_progress:.1f}%) "
                                f"step {global_step} "
                                f"loss={loss.item():.4f} avg_loss={epoch_avg_so_far:.4f}",
                                flush=True,
                            )

                    current_chunk_idx += 1
                    current_chunk_data = next_chunk_data
                    current_dataset_future = next_dataset_future

            if epoch_batches == 0:
                raise ValueError("No training samples were produced from the HF dataset.")
            epoch_avg_loss = epoch_loss_sum / epoch_batches
            print(
                f"epoch {epoch}/{cli_args.epochs} avg_loss={epoch_avg_loss:.4f} "
                f"samples={epoch_samples} chunks={epoch_chunks}",
                flush=True,
            )
    except KeyboardInterrupt:
        print(
            f"\nTraining interrupted at step {global_step}. Saving checkpoint...",
            flush=True,
        )
        save_checkpoint(
            cli_args.checkpoint,
            model,
            optimizer,
            model_args,
            tokenizer_name,
            global_step=global_step,
            epoch=last_epoch,
        )
        print(f"checkpoint saved: {cli_args.checkpoint}", flush=True)
        return

    save_checkpoint(
        cli_args.checkpoint,
        model,
        optimizer,
        model_args,
        tokenizer_name,
        global_step=global_step,
        epoch=last_epoch,
    )
    print(f"checkpoint saved: {cli_args.checkpoint}")


def run_infer(cli_args):
    device = torch.device(cli_args.device)
    model, tokenizer = load_checkpoint(cli_args.checkpoint, device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    encoder_idx = tokenizer(
        cli_args.text,
        return_tensors="pt",
        max_length=model.args.block_size,
        truncation=True,
        padding="max_length",
    ).input_ids.to(device)
    encoder_k_pad_mask = encoder_idx.eq(pad_token_id)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.cls_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.sep_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.pad_token_id
    if bos_token_id is None:
        bos_token_id = 0
    decoder_start = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)
    generated_decoder = generate_greedy(
        model,
        encoder_idx,
        decoder_start,
        cli_args.max_new_tokens,
        eos_token_id=tokenizer.sep_token_id,
        encoder_k_pad_mask=encoder_k_pad_mask,
    )
    print(tokenizer.decode(generated_decoder[0, 1:], skip_special_tokens=True))


def main():
    cli_args = parse_cli_args()
    if cli_args.mode == "train":
        run_train(cli_args)
    else:
        run_infer(cli_args)


if __name__ == "__main__":
    print("start")
    main()
