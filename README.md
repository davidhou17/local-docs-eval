# docs_eval -- Local LLM Documentation Q&A Evaluation

Evaluates how well local LLMs answer documentation questions by scoring responses against a golden dataset using an LLM-as-judge. Uses RAG over your local docs for grounded answers.

**Zero external dependencies** -- only requires Python 3.10+ and [Ollama](https://ollama.com).

## Quick start

### Step 1: Install Ollama

```bash
brew install ollama
```

Or download from https://ollama.com/download.

### Step 2: Start the Ollama server

```bash
ollama serve
```

Leave this running in a separate terminal.

### Step 3: Pull models

```bash
ollama pull mxbai-embed-large  # embeddings for RAG retrieval
ollama pull qwen3:8b           # judge model + model to evaluate
```

To compare multiple models, also pull:

```bash
ollama pull gemma3:12b
ollama pull mistral
ollama pull llama3.1:8b
```

> **Memory guide (Apple Silicon):** On 32 GB machines, the judge + eval model +
> embedding model all need to fit in memory simultaneously. The defaults above
> peak at ~14 GB. Avoid 70B+ models (e.g. `llama3.3`) — they require 40 GB+
> and will fall back to slow CPU offloading.
>
> For best performance, set `OLLAMA_MAX_LOADED_MODELS=3` so Ollama keeps all
> three models resident instead of swapping:
>
> ```bash
> export OLLAMA_MAX_LOADED_MODELS=3
> ```

### Step 4: Run a quick test

Clone this repo and run from the parent directory of `docs_eval_oss/`, pointing `--docs-dir` at your `.mdx` documentation:

```bash
python3 -m docs_eval_oss --docs-dir /path/to/your/docs --limit 3
```

The first run builds an embedding index over the docs (~1-2 min). This is cached automatically -- subsequent runs start instantly.

You should see output like:

```
  Building RAG index from /path/to/your/docs ...
  Loaded 42 documents
  Created 860 chunks, embedding with mxbai-embed-large ...
  Index cached to .rag_index_mxbai-embed-large.json

Evaluating model: qwen3:8b (judge: qwen3:8b, responder: rag)
--------------------------------------------------
  [1/3] Q1 score=1.0
  [2/3] Q2 score=0.75
  [3/3] Q3 score=1.0

--- qwen3:8b ---
  Mean accuracy: 0.917
  Distribution:  1.0=2  0.75=1  0.0=0
  By category:
    ...
```

### Step 5: Run the full evaluation

```bash
python3 -m docs_eval_oss --docs-dir /path/to/your/docs
```

This runs all questions in your CSV. Takes ~10-30 min depending on your hardware.

### Step 6: Compare multiple models

```bash
python3 -m docs_eval_oss --docs-dir /path/to/your/docs --models qwen3:8b,gemma3:12b,mistral,llama3.1:8b
```

Prints a comparison table at the end showing each model's score overall and by category.

### Step 7 (optional): Save results to CSV

```bash
python3 -m docs_eval_oss --docs-dir /path/to/your/docs --output results.csv
```

The CSV contains per-question scores, judge reasoning, and model responses for every model.

## How it works

```
  docs/*.mdx                golden-dataset-docs-q-and-a.csv
       |                                |
       v                                v
  [Chunk + Embed]                [Load questions]
  (mxbai-embed-large)                  |
       |                                |
       v                                v
  Vector index              For each question + model:
  (cached as JSON)            1. Embed question
       |                      2. Retrieve top-k doc chunks (cosine sim)
       |                      3. Stuff into prompt + send to model
       |                      4. Judge response vs golden answer
       v                                |
  .rag_index_*.json                     v
                              Aggregated scores + comparison table
```

1. **Index**: All `.mdx` files are chunked (~500 chars, overlapping) and embedded via Ollama's `/api/embed` endpoint. The index is cached as a JSON file in the docs directory -- subsequent runs skip re-embedding unless the docs change.
2. **Retrieve**: For each question, the query is embedded and the top-k most similar chunks are retrieved via cosine similarity.
3. **Generate**: The retrieved chunks + question are sent to the model under test.
4. **Judge**: An LLM judge (qwen3:8b) compares the response to the golden answer:
   - **1.0** -- Correct and complete
   - **0.75** -- Partially correct, key info present but incomplete
   - **0.0** -- Wrong, irrelevant, or missing key information
5. **Report**: Per-model mean score, by-category breakdown, and (for multi-model runs) a comparison table.

## CLI flags

| Flag             | Default                              | Description                                      |
|------------------|--------------------------------------|--------------------------------------------------|
| `--docs-dir`     | auto-detect sibling `docs/`          | Path to docs directory with `.mdx` files         |
| `--models`       | `qwen3:8b`                           | Comma-separated Ollama model names to evaluate   |
| `--judge-model`  | `qwen3:8b`                           | Ollama model used as the LLM judge               |
| `--embed-model`  | `mxbai-embed-large`                  | Ollama model for RAG embeddings                  |
| `--top-k`        | `5`                                  | Number of doc chunks to retrieve per question    |
| `--responder`    | `rag`                                | Responder type: `rag`, `naive_baseline`, or `golden_answer` |
| `--csv`          | bundled `golden-dataset-docs-q-and-a.csv` | Path to golden dataset CSV                  |
| `--limit`        | all rows                             | Evaluate only the first N rows                   |
| `--output`       | none                                 | Write per-row results CSV                        |

## Environment variables

| Variable          | Default                    | Description              |
|-------------------|----------------------------|--------------------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434`   | Ollama server URL        |
| `GOLDEN_DATASET_CSV` | bundled CSV             | Default CSV path         |

## Bringing your own dataset

The bundled `golden-dataset-docs-q-and-a.csv` is a small example. For real evaluations, create your own CSV with these columns:

| Column | Description |
|---|---|
| `Question_ID` | Unique identifier for the question (e.g. `1`, `2`, ...) |
| `Category` | Category label for grouping results (e.g. `Getting Started`, `Auth`) |
| `Question` | The question to ask the model |
| `Answer` | The expected (golden) answer |

Point the tool at your CSV with `--csv /path/to/your-dataset.csv`.

## Responders

- **rag** (default) -- Retrieves relevant doc chunks via embedding similarity, then asks the LLM to answer grounded in that context. This is the main evaluation mode.
- **naive_baseline** -- Sends the question to the LLM with no context. Measures raw model knowledge as a baseline.
- **golden_answer** -- Returns the expected answer as-is. Use to sanity-check the judge pipeline (expect mean score ~1.0).
