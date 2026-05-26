# Automatic Log Template Extraction Using Large Language Models

Benchmarks 4 pretrained LLMs (GPT-2, InCoder, T5, BART) on structured 
log template extraction using In-Context Learning (ICL) and 
PEFT-based prefix tuning. Evaluates outputs with ROUGE-1/2/L, BLEU, 
PA, PTA, and RTA metrics across multiple log datasets.

## Problem

System logs are unstructured — extracting reusable templates from raw 
log messages manually doesn't scale. This project applies LLMs to 
automate log template extraction (log parsing), replacing hand-crafted 
rules and similarity thresholds used by traditional tools like Drain.

## Approach

Benchmarks 4 pretrained LLMs (GPT-2, InCoder, T5, BART) on structured 
log template extraction using two techniques:
- **ICL (In-Context Learning)** — few-shot prompting across all 4 models
- **Prefix Tuning (PT)** — PEFT-based fine-tuning on T5 and BART

## Datasets

Evaluated on standard log parsing benchmarks including HDFS, BGL, 
and other log datasets from the LogHub collection.

## Metrics

ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Parsing Accuracy (PA), 
Perfect Template Accuracy (PTA), Root Template Accuracy (RTA)

## Results

Results stored in:
- `out/ICL/predictions` and `out/ICL/results` — ICL run outputs
- `PT_results/` — prefix tuning outputs

## Requirements

    Python 3.6 or later
    Hugging Face Transformers library
    nltk
    pandas
    torch
    transformers
    rouge-metric
    accelerate
    sentencepiece

Install the required dependencies using the following command:

pip install -r requirements.txt

Generate ICL Results

To generate model predictions using ICL and evaluate the results, run the script:

python icl.py

This script will download the pretrained models gpt-2, incoder, t5 and bart. Then, generates the predictions of the test data using ICL for each model and saves these predictions in out/ICL/predictions. It the evaluates each model, measuring Rouge-1, Rouge-2, Rouge-L, BLEU, PA, PTA and RTA metrics for each dataset and saves these results inside out/ICL/results folder.

To evaluate only one of these models, pass the model name as the input argument:

python icl.py gpt-2

python icl.py incoder

python icl.py t5

python icl.py bart
Generate Prefix Tuning (PT) Results
Requirements
install Hugging Face Libraries

pip install peft

pip install "transformers==4.27.1" datasets accelerate evaluate bitsandbytes loralib --upgrade --quiet
install additional dependencies needed for training

pip install rouge-score tensorboard py7zr

To generate model predictions using PT with T5 model on all four log-based datasets and evaluate the results, run the script:

python pt.py t5

Similarly for bart model:

python pt.py bart

This script will download the pretrained models t5 and bart. Then, generates the predictions on the test data using PT for each model and saves these predictions with results in PT_results directory. It the evaluates each model using the same evaluation metrics as adopted for ICL.
