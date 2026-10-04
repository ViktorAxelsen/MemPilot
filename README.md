# MemPilot: Orchestrating On-Demand Multimodal Memory Curation for LLM Agents

[News](#news) · [Installation](#installation) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

<a id="overview"></a>

## 🧠 Overview

**MemPilot** learns how to gather and process multimodal memory at query time.
A policy chooses what evidence to access, which LLM/VLM to use, when images are
needed, and when to stop gathering evidence and answer.

- **Two memory actions:** RETRIEVE searches a prepared memory bank; CURATE asks a
  selected LLM/VLM to process evidence from the original history.
- **Resource-aware training:** GDPO combines answer quality, cost and latency
  preferences, with marginal utility credit for memory-gathering stages.
- **Joint training and transfer evaluation:** train on Mem-Gallery,
  WorldMemArena-Lifelong and H2HMem-Dyadic; evaluate on these benchmarks plus
  MemEye Open and MEMLENS.

The default memory bank is built offline with LLMLingua-2. An existing bank from
another system can replace it through the [memory adapter](#using-an-existing-memory-bank).

<a id="news"></a>

## 📰 News

- 🚀 **[2026-10-05]** We release **MemPilot**, our framework for on-demand multimodal
  memory curation.

<a id="installation"></a>

## 🚀 Installation

Run commands from the repository root. The environment targets **Linux x86_64,
Python 3.12 and NVIDIA GPUs**.

```bash
bash scripts/setup_environment.sh --env-name mempilot
conda activate mempilot
```

For an existing Python 3.12 environment:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-flash-attn.txt
python -m pip check
```

The setup uses verl 0.8.0, PyTorch 2.8 and vLLM 0.11.0. The supplied FlashAttention
wheel is for Python 3.12 and PyTorch 2.8 on Linux x86_64.

<a id="configuration"></a>

## ⚙️ Configuration

Edit training settings in [`scripts/train.sh`](scripts/train.sh) and evaluation
settings in [`scripts/eval_common.sh`](scripts/eval_common.sh). Set credentials
in the environment, or uncomment and fill in the `export` examples in those scripts:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export HUGGING_FACE_HUB_TOKEN="your_hugging_face_token"
export WANDB_API_KEY="your_wandb_api_key"
```

Script exports override environment values. Answer replacement and judging can
use separate `ANSWER_API_KEY` / `ANSWER_BASE_URL` and `JUDGE_API_KEY` /
`JUDGE_BASE_URL` settings. Their Python CLIs also accept OpenAI-compatible
`OPENAI_API_KEY` / `OPENAI_BASE_URL` settings.

Configure the CURATE model pool and resource profiles in
[`configs/runtime_memory_tool.yaml`](configs/runtime_memory_tool.yaml).

<a id="preparing-data"></a>

## 📦 Preparing data

Converters download their datasets from Hugging Face by default. Use
`--dataset_dir` for local data and `--help` for preprocessing options.
They preserve raw history and images, prepare LLMLingua-2 memory, and build
separate dense indexes for runtime retrieval.

Prepare and merge the three training benchmarks:

```bash
python -m data.prepare_mem_gallery_verl --output_dir data/mem_gallery
python -m data.prepare_worldmemarena_verl \
  --regime lifelong --output_dir data/worldmemarena_lifelong
python -m data.prepare_h2hmem_verl \
  --variant dyadic --output_dir data/h2hmem_dyadic

python -m data.merge_multimodal_verl \
  --input_dir data/mem_gallery \
  --input_dir data/worldmemarena_lifelong \
  --input_dir data/h2hmem_dyadic \
  --output_dir data/multimodal_unified
```

Each training benchmark has conversation/world-disjoint `train.parquet`,
`val.parquet` and `test.parquet` splits (70/10/20). WorldMemArena preserves the
history visible at each question's checkpoint.

Prepare the transfer benchmarks:

```bash
python -m data.prepare_memeye_verl --output_dir data/memeye
python -m data.prepare_memlens_verl \
  --context_length 32k --evaluation_subset agent --output_dir data/memlens_32k_agent
```

<a id="training"></a>

## 🏋️ Training

Set the model, GPUs, resource weights and training parameters in `scripts/train.sh`:

```bash
bash scripts/train.sh
```

Training uses the mixed data in `data/multimodal_unified`, with `val.parquet` for
periodic validation. Change `DATA_DIR` in the script or export it to select another
prepared directory. Additional arguments are forwarded to verl/Hydra:

```bash
DATA_DIR=/path/to/prepared_data bash scripts/train.sh trainer.total_epochs=5
```

Checkpoints are saved under `checkpoints/<project>/<run_name>/`; console and
model-call logs are saved under `logs/`. Set `LOGGER='["console"]'` in the script
to disable W&B logging.

<a id="evaluation"></a>

## 🧪 Evaluation

Configure the policy model, GPUs, answer model and judge in `scripts/eval_common.sh`.
`MODEL_PATH` must match the trained policy. By default, answer replacement uses
local `Qwen/Qwen3-VL-4B-Instruct`; judging uses OpenRouter GPT-4o-mini.

Run all five datasets:

```bash
bash scripts/eval.sh \
  --checkpoint-path /path/to/global_step_100 \
  --output-dir eval_outputs/run1
```

Add `--datasets mem_gallery memeye` to evaluate only selected benchmarks.
All evaluation scripts accept the same dataset names and output directory:

| Dataset name | Default test file under `data/` | Path override |
| --- | --- | --- |
| `mem_gallery` | `mem_gallery/test.parquet` | `MEM_GALLERY_TEST_FILE` |
| `worldmemarena` | `worldmemarena_lifelong/test.parquet` | `WORLDMEMARENA_TEST_FILE` |
| `h2hmem` | `h2hmem_dyadic/test.parquet` | `H2HMEM_TEST_FILE` |
| `memeye` | `memeye/open/test.parquet` | `MEMEYE_TEST_FILE` |
| `memlens` | `memlens_32k_agent/test.parquet` | `MEMLENS_TEST_FILE` |

`eval.sh` runs **policy inference → answer replacement → final metrics**.
The same stages can be run separately:

```bash
bash scripts/eval_policy.sh --checkpoint-path /path/to/global_step_100 \
  --output-dir eval_outputs/run2 --datasets mem_gallery memeye
bash scripts/eval_answer_replacement.sh \
  --output-dir eval_outputs/run2 --datasets mem_gallery memeye
bash scripts/eval_metrics.sh \
  --output-dir eval_outputs/run2 --datasets mem_gallery memeye
```

Each dataset's outputs are saved in `policy/`, `answer_replacement/`, `metrics/`
and `logs/` below the selected output directory. Final reports include F1,
judge scores, API cost and proxy latency. Use a new output directory for
each evaluation configuration. Interrupted answer replacement or judging can
reuse its progress cache when that stage is rerun; policy inference does not resume.

Set `ANSWER_BACKEND=openai`, `ANSWER_MODEL` and `ANSWER_BASE_URL` for API-based
answer replacement. Custom answer models also need a resource profile in the YAML.

<a id="using-an-existing-memory-bank"></a>

## 🔌 Using an existing memory bank

Choose **either the default LLMLingua bank or an external bank** through the
prepared data path. The adapter replaces the RETRIEVE corpus and its index;
original history and images remain available to CURATE.

Provide a JSON/JSONL record for each QA scope in the input splits:

```json
{"data_source": "mem_gallery", "conversation_id": "conversation-1", "question_id": "question-1", "memories": [{"id": "fact-1", "text": "Alice visited Rome."}]}
```

Copy identifiers from the prepared rows and include only evidence available at
that question's checkpoint. `memories` accepts strings or `id`/`text` records.
For other formats, customize `convert_memory_items` in
[`data/convert_memory_bank.py`](data/convert_memory_bank.py); runtime code uses
the same normalized bank format.

For joint training, convert the merged data and select its output directory:

```bash
python -m data.convert_memory_bank \
  --input_dir data/multimodal_unified \
  --memory_file /path/to/external_memory.jsonl \
  --output_dir data/multimodal_external

DATA_DIR=data/multimodal_external bash scripts/train.sh
```

For evaluation, convert each benchmark directory and point its `*_TEST_FILE`
setting at the resulting `test.parquet`. No memory-mode switch is needed.

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

We thank the [verl](https://github.com/verl-project/verl) team for their open-source
RL training framework, and the authors of
**[Mem-Gallery](https://github.com/YuanchenBei/Mem-Gallery)**,
**[WorldMemArena](https://github.com/UCSB-AI/WorldMemArena)**,
**[H2HMem](https://github.com/varib1/H2HMEM)**,
**[MemEye](https://github.com/MinghoKwok/MemEye)**, and
**[MEMLENS](https://github.com/xrenaf/MEMLENS)** for their benchmarks and evaluation resources.

<a id="citation"></a>

## 📚 Citation

If you find MemPilot useful in your research, please consider citing:

```bibtex
@misc{zhang2026mempilot,
  title  = {MemPilot: Orchestrating On-Demand Multimodal Memory Curation for LLM Agents},
  author = {Haozhen Zhang and Haodong Yue and Quanyu Long and Jianzhu Bao and
            Qingyuan Liu and Tao Feng and Bohan Liu and Weida Liang and Wenya Wang},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/ViktorAxelsen/MemPilot}
}
```

*arXiv identifier coming soon.*
