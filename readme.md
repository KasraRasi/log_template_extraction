# Automatic Log Template Extraction Using Large Language Models

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

