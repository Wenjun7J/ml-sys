# ml-sys

Machine learning systems experiments and small model implementations.

The current repository focuses on a six-layer encoder-decoder Transformer
inspired by "Attention Is All You Need". It trains on Hugging Face translation
data [wmt18](https://huggingface.co/datasets/wmt/wmt18), saves PyTorch checkpoints, and runs greedy decoding for inference. The
default setup targets `zh -> en` translation with a multilingual BERT
tokenizer.

## Projects

- [Transformer](#transformer): six-layer encoder-decoder Transformer for
  translation experiments, with chunked Hugging Face dataset loading,
  checkpoint save and resume support, and greedy-decoding inference

## Transformer

A six-layer encoder-decoder Transformer demo following "Attention Is All You
Need".

The current entrypoint is `models/transformer/transformer.py`. It trains on a
Hugging Face translation dataset, saves PyTorch checkpoints with `torch.save`,
and runs greedy decoding for inference.

### Files

- `models/transformer/transformer.py`: current training and inference
  entrypoint.

### Requirements

The project dependency file is `models/requirements.txt`.

### Setup

Run these commands from the project root to create a local virtual environment
and install the required Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r models/requirements.txt
```

### Quick Start

Run these commands from the project root after activating `.venv`.

Show CLI options:

```bash
python models/transformer/transformer.py --help
```

Run a small training smoke test:

```bash
python models/transformer/transformer.py \
  --mode train \
  --checkpoint transformer.pt \
  --device cuda \
  --epochs 10 \
  --max_samples 128 \
  --chunk_lines 128 \
  --batch_size 32 \
  --tokenize_workers 4 \
  --optimizer adam \
  --shuffle_dataset
```

Run inference:

```bash
python models/transformer/transformer.py \
  --mode infer \
  --checkpoint transformer.pt \
  --device cuda \
  --text "我喜欢学习大语言模型。"
```

Run inference with beam search:

```bash
python models/transformer/transformer.py \
  --mode infer \
  --checkpoint transformer.pt \
  --device cuda \
  --text "我喜欢学习大语言模型。" \
  --decode_strategy beam \
  --beam_size 4
```

### Defaults

Training defaults:

| Option | Default | Notes |
| --- | --- | --- |
| `--mode` | `train` | Use `infer` for checkpoint inference. |
| `--checkpoint` | `transformer.pt` | Save path in train mode, load path in infer mode. |
| `--tokenizer_name` | `bert-base-multilingual-cased` | Passed to `AutoTokenizer.from_pretrained`. |
| `--hf_dataset_name` | `wmt/wmt18` | Hugging Face dataset name. |
| `--hf_dataset_config` | `zh-en` | Dataset config. |
| `--hf_dataset_split` | `train` | Dataset split. |
| `--src_lang` | `zh` | Source key inside each `translation` record. |
| `--tgt_lang` | `en` | Target key inside each `translation` record. |
| `--epochs` | `8` | Number of training epochs. |
| `--batch_size` | `16` | PyTorch `DataLoader` batch size. |
| `--optimizer` | `adam` | `adamw` is available but not the recommended default here. |
| `--lr` | `3e-4` | Learning rate. |
| `--log_interval` | `100` | Print train progress every N batches. |
| `--max_samples` | `0` | `0` means no sample limit. |
| `--chunk_lines` | `200000` | Number of translation pairs per dataset chunk. |
| `--data_offset` | `0` | Number of dataset rows to skip before reading samples. |
| `--shuffle_dataset` | disabled | Globally shuffle before chunking when enabled. |
| `--shuffle_seed` | `42` | Each epoch uses `shuffle_seed + epoch - 1`. |
| `--tokenize_batch_size` | `2000` | Tokenization mini-batch size inside each chunk. |
| `--tokenize_workers` | `16` | Number of tokenization worker threads. |
| `--tokenize_log_interval` | `10` | Print tokenization progress every N tasks. |
| `--max_length` | `128` | Training and inference sequence length. |
| `--device` | `cuda` if available, otherwise `cpu` | Runtime device. |

Model defaults when training without `--resume_from_checkpoint`:

| Field | Default |
| --- | --- |
| `n_layer` | `6` |
| `n_embd` | `512` |
| `n_heads` | `8` |
| `dropout` | `0.1` |
| `max_length` | CLI `--max_length` |
| `vocab_size` | tokenizer vocab size |

Inference defaults:

| Option | Default | Notes |
| --- | --- | --- |
| `--text` | `I love learning large language models.` | Override this for the source language used during training. |
| `--max_new_tokens` | `20` | Maximum number of generated decoder tokens. |
| `--decode_strategy` | `greedy` | Use `beam` to enable beam search decoding. |
| `--beam_size` | `4` | Number of beams kept when `--decode_strategy beam` is used. |
| `--beam_length_penalty` | `0.0` | Length penalty exponent for beam ranking; `0.0` disables length normalization. |
| `--print_encoder_out` | disabled | Prints encoder tensor diagnostics before decoding. |

### Training Behavior

- Loads translation data from Hugging Face in chunks instead of keeping the
  full dataset in memory.
- Extracts `src_lang` and `tgt_lang` from each row's `translation` object.
- Tokenizes each chunk in parallel with a thread pool.
- Builds a `DataLoader` per chunk and shuffles batches inside each chunk.
- Use `--shuffle_dataset` to shuffle the full Hugging Face dataset before
  chunking. This can reduce loss jumps caused by source-order domain shifts
  between chunks.
- Logs `next_loss`, `first_loss`, and running average loss.
- Saves a checkpoint in `finally`, so completed runs, interrupted runs, and
  failed runs all attempt to write the latest model weights.

Tokenization workers load the tokenizer with `local_files_only=True` after the
main process initializes it. Make sure the tokenizer can be resolved or cached
before using many workers.

### Checkpoints

The checkpoint contains:

- `model_state_dict`
- `model_args`
- `tokenizer_name`

The checkpoint does not contain:

- optimizer state
- learning rate scheduler state
- epoch index
- global step
- dataloader, chunk, or shuffle state

`--resume_from_checkpoint` restores model weights and `model_args`, but it is
not a full training-state resume. After resuming, the optimizer is recreated,
epochs restart from `1`, logging starts from `step 0`, and the loss may jump
for a while.

When a checkpoint stores `tokenizer_name`, train resume and inference use that
tokenizer even if the CLI passes a different `--tokenizer_name`. If checkpoint
`model_args.max_length` differs from CLI `--max_length`, the checkpoint value is
used.

### Inference

Inference loads `--checkpoint`, tokenizes `--text`, and decodes from a single
BOS token until EOS or `--max_new_tokens`. Greedy decoding is the default; pass
`--decode_strategy beam` to use beam search.

The default training direction is `zh -> en`, so use Chinese source text for
checkpoints trained with the default dataset and language settings.

### Known Limitations

- No sampling decoder.
- No validation loop, metrics, or checkpoint selection.
- Resume restores weights only, not optimizer or dataloader state.
