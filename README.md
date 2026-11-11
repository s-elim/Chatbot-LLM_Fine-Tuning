# Chatbot-LLM_Fine-Tuning

A single Jupyter notebook that builds a GPT-2 style language model in PyTorch, then runs the fine-tuning loop on an instruction dataset. It demonstrates how fine-tuning can improve LLM performance while following instructions.

## Contents

`Fine-tuned LLM Evaluation.ipynb` (889 cells, Python 3.12 kernel):

1. **Text to tokens:** tokenization, byte pair encoding with `tiktoken`, data loader, token and positional embeddings.
2. **Attention:** self attention, causal attention, multi-head attention.
3. **GPT model from scratch:** layer normalization, GELU feed-forward block, shortcut connections, transformer block, text generation.
4. **Training:** cross-entropy loss and perplexity, training loop, temperature scaling and top-k sampling, saving and loading weights.
5. **Pretrained weights:** loading GPT-2 weights from OpenAI.
6. **Fine-tuning for classification:** SMS spam classifier.
7. **Instruction fine-tuning:** Alpaca-format prompts from `instruction-data.json`, train, validation and test split, fine-tuning, saving the responses.
8. **Evaluation:** scoring the fine-tuned model's responses from 0 to 100 with `llama3` served by a local Ollama instance.

## Requirements

The repository has no requirements file. The notebook imports:

```bash
pip install jupyter torch tiktoken numpy matplotlib pandas tensorflow tqdm psutil
```

It also needs:

- `the-verdict.txt` in the working directory (the first cells read it).
- A `gpt_download3` module that provides `download_and_load_gpt2`. It is not in this repository.
- A running Ollama server with the `llama3` model for the evaluation step (`http://localhost:11434/api/chat`).

The notebook downloads the SMS spam dataset and `instruction-data.json` itself.

## Usage

```bash
jupyter notebook "Fine-tuned LLM Evaluation.ipynb"
```

Run the cells in order. Later sections reuse classes and functions defined earlier.

## License

This repository has no LICENSE file.
