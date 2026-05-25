# Agentic LLM Pipeline — Automatic Log Template Extraction

A multi-step agentic LLM evaluation pipeline for structured output 
extraction from unstructured log data, using in-context learning (ICL) 
and PEFT-based fine-tuning across multiple pretrained models.

## Overview
Unstructured system logs contain critical operational patterns that are 
hard to extract reliably. This project builds an agentic, multi-step 
LLM pipeline that extracts structured log templates from raw log text — 
benchmarking four pretrained models with standardized evaluation metrics 
and reproducible experiment storage.

## Key Features
- **Agentic multi-step pipeline** — document preprocessing → prompt 
  construction → LLM generation → structured output extraction → 
  evaluation, with state passed between steps
- **4 pretrained models benchmarked** — GPT-2, T5, BART, InCoder 
  evaluated under consistent conditions
- **In-context learning (ICL)** — few-shot prompt engineering to guide 
  structured output generation without full fine-tuning
- **PEFT-based prefix tuning** — parameter-efficient fine-tuning applied 
  to optimize structured extraction quality with minimal compute
- **Standardized LLMOps evaluation** — ROUGE-1, ROUGE-2, ROUGE-L, BLEU, 
  Parsing Accuracy (PA), Perfect Template Accuracy (PTA), 
  Root Template Accuracy (RTA)
- **Reproducible experiment storage** — all model outputs stored as 
  structured JSON for downstream comparison and analysis
- **RAG-adjacent retrieval** — source document chunking and context 
  injection strategies to improve LLM grounding on long log files

## Models
| Model   | Approach         | Tuning        |
|---------|-----------------|---------------|
| GPT-2   | ICL few-shot    | Prefix tuning |
| T5      | ICL few-shot    | Prefix tuning |
| BART    | ICL few-shot    | Prefix tuning |
| InCoder | ICL few-shot    | Prefix tuning |

## Tech Stack
Python · Hugging Face Transformers · PyTorch · PEFT · LangChain · 
pandas · ROUGE · BLEU · JSON structured output storage

### Requirements

- Python 3.6 or later
- [Hugging Face Transformers](https://github.com/huggingface/transformers) library
- nltk
- pandas
- torch 
- transformers 
- rouge-metric 
- accelerate 
- sentencepiece

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Generate ICL Results

To generate model predictions using ICL and evaluate the results, run the script:

`python icl.py`

This script will download the pretrained models `gpt-2`, `incoder`, `t5` and `bart`. Then, generates the predictions of the test data using ICL for each model and saves these predictions in `out/ICL/predictions`. It the evaluates
each model, measuring `Rouge-1`, `Rouge-2`, `Rouge-L`, `BLEU`, `PA`, `PTA` and `RTA` metrics for each dataset and saves these results inside `out/ICL/results` folder.

To evaluate only one of these models, pass the model name as the input argument:

`python icl.py gpt-2`

`python icl.py incoder`

`python icl.py t5`

`python icl.py bart`



### Generate Prefix Tuning (PT) Results

### Requirements
#### install Hugging Face Libraries
`pip install peft`

`pip install "transformers==4.27.1" datasets accelerate evaluate bitsandbytes loralib --upgrade --quiet`
#### install additional dependencies needed for training
`pip install rouge-score tensorboard py7zr`

To generate model predictions using PT with T5 model on all four log-based datasets and evaluate the results, run the script:

`python pt.py t5`

Similarly for bart model:

`python pt.py bart`

This script will download the pretrained models `t5` and `bart`. Then, generates the predictions on the test data using PT for each model and saves these predictions with results in `PT_results` directory. It the evaluates
each model using the same evaluation metrics as adopted for ICL.

