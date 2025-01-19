
# -*- coding: utf-8 -*-

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, BartTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPT2LMHeadModel
from datasets import concatenate_datasets
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import BartForConditionalGeneration, BartTokenizer
from peft import PrefixTuningConfig
from peft import get_peft_model, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import sys
import os


"""
# install Hugging Face Libraries
!pip install peft
!pip install "transformers==4.27.1" datasets accelerate evaluate bitsandbytes loralib --upgrade --quiet
# install additional dependencies needed for training
!pip install rouge-score tensorboard py7zr
"""


def compute_rouge(model, tokenizer, path_to_data):
  # Metric
  metric = evaluate.load("rouge")

  def evaluate_peft_model(sample,max_target_length=50):
      # generate summary
      outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)
      prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
      # decode eval sample
      # Replace -100 in the labels as we can't decode them.
      labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
      labels = tokenizer.decode(labels, skip_special_tokens=True)

      # Some simple post-processing
      return prediction, labels

  # # load test dataset from distk
  test_dataset = load_from_disk(path_to_data).with_format("torch")   # data/eval

  # run predictions
  predictions, references = [] , []
  for sample in tqdm(test_dataset):
      p,l = evaluate_peft_model(sample)
      predictions.append(p)
      references.append(l)

  # compute metric
  rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

  # print results
  print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
  print(f"rouge2: {rogue['rouge2']* 100:2f}%")
  print(f"rougeL: {rogue['rougeL']* 100:2f}%")
  print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")
  return predictions, references, rogue



def load_tokenizer(model_name):
  print("model_name", model_name)
  if model_name == "t5":
    print("load google/flan-t5-xxl")
    model_id="google/flan-t5-xxl"
    # Load tokenizer of FLAN-t5-XL
    tokenizer = AutoTokenizer.from_pretrained(model_id)

  elif model_name == "bart" or model_name == "bart-large":
    print("loading tokenizer: facebook/bart-large")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

  elif model_name == "bart-large-xsum":
    print("loading tokenizer: facebook/bart-large-xsum")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")

  # elif model_name == "gpt2":
  #   # GPT-2 tokenizer
  #   tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

  # elif model_name == "incoder-1B":
  #   tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
  return tokenizer


def load_model(model_name):
  # # huggingface hub model id
  # # model_id = "philschmid/flan-t5-xxl-sharded-fp16"
  if model_name == "t5":
    model_id = "philschmid/flan-t5-base-samsum"
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)#, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)

  # # 2nd Model: BART
  elif model_name == "bart-large":
    print("loading bart-large...")
    # model_id = "philschmid/bart-base-samsum"
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

  elif model_name == "bart-large-xsum":
    print(f"loading {model_name}...")
    model_id = "facebook/bart-large-xsum"
    model = BartForConditionalGeneration.from_pretrained(model_id, forced_bos_token_id=0)


  # elif model_name == "gpt2":
  #   print("loading gpt2-xl...")
  #   model_id = "gpt2-xl"
  #   model = GPT2LMHeadModel.from_pretrained(model_id)
  # elif model_name == "incoder-1B":
  #   model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")
  return model



def load_dataset(dataset_name, path_to_csv):
  """### Load custom dataset"""
  # # Read your custom dataset from CSV
  df = pd.read_csv(path_to_csv)

  """### Preprocessing"""

  # Preprocessing
  custom_data = df[['Content', 'EventId', 'EventTemplate']]   # load only relevant columns
  # Convert your custom dataset to a format compatible with datasets library
  custom_dataset = Dataset.from_pandas(custom_data)

  # Rename columns if needed
  custom_dataset = custom_dataset.rename_column("Content", "dialogue")
  custom_dataset = custom_dataset.rename_column("EventTemplate", "summary")
  custom_dataset = custom_dataset.rename_column("EventId", "id")

  # Split your custom dataset into train and validation sets
  dataset = custom_dataset.train_test_split(test_size=0.1)


  #thunderbird
  print(custom_data.head())

  print(f"Train dataset size: {len(dataset['train'])}")
  print(f"Test dataset size: {len(dataset['test'])}")
  return dataset


def preprocess(dataset, model_name, tokenizer, prefix_keyword="summarize:"):
  """Before we can start training, we need to preprocess our data. 
  Abstractive Summarization is a text-generation task. Our model will take a text as input and 
  generate a summary as output. We want to understand how long our input and output will take to batch our data efficiently."""

  # The maximum total input sequence length after tokenization.
  # Sequences longer than this will be truncated, sequences shorter will be padded.
  tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], 
                                                                                                       truncation=True), batched=True, remove_columns=["dialogue", "summary"])
  input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
  # take 85 percentile of max length for better utilization
  max_source_length = int(np.percentile(input_lenghts, 85))
  print(f"Max source length: {max_source_length}")

  # The maximum total sequence length for target text after tokenization.
  # Sequences longer than this will be truncated, sequences shorter will be padded."
  tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], 
                                                                                                        truncation=True), batched=True, remove_columns=["dialogue", "summary"])
  target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
  # take 90 percentile of max length for better utilization
  max_target_length = int(np.percentile(target_lenghts, 90))
  print(f"Max target length: {max_target_length}")


  def preprocess_function(sample,padding="max_length"):
      # add prefix to the input for t5
      inputs = ["summarize: " + item for item in sample["dialogue"]]   # TODO: change "summarize:" to "log2template:" or "extract_template" or "log parse" later

      # Impact of Domain-specific prefix terms:
      # =======================================
      # We studied the impact of different task specific prefix words such as "extract_template"
      # However, we didn't observe any improvement in the results. For example, when evaluated on BGL dataset
      # we noticed a BLEU score of 0.5 with "summarize" whereas 0.43 was obtained with prefixes ["extract_template", "log2template"].
      # Example:
      # Dataset prefix_keyword Rouge1     Rouge2      RougeL      BLEU
      # BGL 	log2template: 	84.741992 	75.084629 	84.377867 	0.43

      # This indicates that domain specific keywords not necessarily lead to optimal results.
      # Consequently, we adopted the more general prefix terms such as "summarize:" to guide the LM extract
      # relevant templates from the log messages.

      # prefix 2:
      # inputs = ["extract_template: " + item for item in sample["dialogue"]]   # TODO: change "summarize:" to "log2template:" or "extract_template" or "log parse" later

      # # prefix 3:
      # inputs = [prefix_keyword+" " + item for item in sample["dialogue"]]

      # tokenize inputs
      if model_name == "gpt2":
        # # model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
        # model_inputs = tokenizer(inputs, truncation=True)
        # # labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, truncation=True)
        # labels = tokenizer(text_target=sample["summary"], truncation=True)

        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
      #   model_inputs = tokenizer(
      #     inputs,
      #     truncation=True,
      #     padding=True,
      #     max_length=max_source_length,
      #     return_tensors="pt"  # Return PyTorch tensors
      # )

        # model_inputs = tokenizer.encode(inputs, return_tensors="pt", add_special_tokens=True)
        model_inputs = tokenizer(inputs, return_tensors='pt', add_special_tokens=True, truncation=True, padding=True)
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
        # model_inputs = {"text": model_inputs,
        #                 "labels": labels}
        # model_inputs["labels"] = labels


      elif model_name == "incoder-1B":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_inputs = tokenizer(
          inputs,
          truncation=True,
          padding=True,
          max_length=max_source_length,
          return_tensors="pt"  # Return PyTorch tensors
      )
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)


      else:
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

      # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
      # padding in the loss.

      if model_name == "gpt2":
        print("using gpt2 tokenizer")
        # model_inputs["labels"] = labels["input_ids"]
      else:
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

      model_inputs["labels"] = labels["input_ids"]
      return model_inputs

  tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
  print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

  # save datasets to disk for later easy loading
  tokenized_dataset["train"].save_to_disk("data/train")
  tokenized_dataset["test"].save_to_disk("data/eval")
  # tokenized_dataset["validation"].save_to_disk("data/validation")
  return tokenized_dataset


def save_results(dataset_name, model_name, predictions, references, rogue):
  print("dataset name:", dataset_name)
  df = pd.DataFrame()
  for pred, ref in zip(predictions, references):
      row = {
          'predictions': pred,
          'references': ref
      }
      df = pd.concat([df, pd.DataFrame([row])])

  ds_names = [dataset_name]
  res = pd.DataFrame()

  for name in ds_names:
      # reference = [[p.split()] for p in references]
      # candidate = [str(p).split() for p in predictions]
      reference = [[p.split()] for p in df['references']]
      candidate = [str(p).split() for p in df['predictions']]
      
      df['correct'] = df['predictions'] == df['references']
      template_df = df.groupby('references').agg({'references': 'count', 'correct': 'sum'})\
          .rename(columns={'references': 'count'})
      correctly_identified = len(template_df[template_df['correct']>0])
      predicted_templates = len(set(df['predictions']))  # all the unique predicted templates
      #

      row = {
          'Dataset': name,
  # #         "Prefix": prefix_keyword,
          'Rouge-1': f"{rogue['rouge1']* 100:2f}",
          'Rouge-2': f"{rogue['rouge2']* 100:2f}",
          'Rouge-L': f"{rogue['rougeL']* 100:2f}",
          'BLEU': f"{corpus_bleu(reference, candidate):.2f}",
          'PA': f"{df['correct'].sum() / len(df):.2f}",
          'PTA': f"{correctly_identified / predicted_templates:.2f}",
          'RTA': f"{correctly_identified / len(template_df):.2f}"

      }
      
      res = pd.concat([res, pd.DataFrame([row])])

  print("Final results:\n", res)
  # Save dataframe
  print("Save predictions and references ...")
  path = "./PT_results"
  os.makedirs(path, exist_ok=True)
  df.to_csv(os.path.join(path, f"{model_name}_{dataset_name}_predictions.csv"))
  print("Save results:")
  res.to_csv(os.path.join(path, f"{model_name}_{dataset_name}_results.csv"))
  
  print("Saved.")


def train(model_name, dataset_name, dataset):

  """To train our model, we need to convert our inputs (text) to token IDs. This is done by a 🤗 Transformers Tokenizer. 
  If you are not sure what this means, check out **[chapter 6](https://huggingface.co/course/chapter6/1?fw=tf)** of the Hugging Face Course."""

  # model_name = "t5"   #
  # model_name = "bart"   #
  # model_name = "bart-large-xsum"   #
  # model_name = "incoder-1B"
  # model_name = "gpt2"   #

  tokenizer = load_tokenizer(model_name)
  tokenized_dataset = preprocess(dataset, model_name, tokenizer, prefix_keyword="summarize:")
  
  """## 3. Fine-Tune T5 with Prefix-tuning
  """

  model = load_model(model_name)

  """### Prefix tuning config"""
  # prefix_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)   # previous experiments with 20 virtual tokens
  prefix_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,
                                    num_virtual_tokens=20
                                    #  num_attention_heads = 12
                                    )   # previous experiments with 20 virtual tokens

  # add prefix-tuning
  model = get_peft_model(model, prefix_config)
  # model.print_trainable_parameters()

 
  # we want to ignore tokenizer pad token in the loss
  label_pad_token_id = -100
  # Data collator
  data_collator = DataCollatorForSeq2Seq(
      tokenizer,
      model=model,
      label_pad_token_id=label_pad_token_id,
      pad_to_multiple_of=8
  )


  """The last step is to define the hyperparameters (`TrainingArguments`) we want to use for our training."""

  
  output_dir=f"prefix-{model_name}"
  num_train_epochs = 5
  # Define training args
  training_args = Seq2SeqTrainingArguments(
      output_dir=output_dir,
      auto_find_batch_size=True,
      learning_rate=1e-3, # higher learning rate
      num_train_epochs=num_train_epochs,
      logging_dir=f"{output_dir}/logs",
      logging_strategy="steps",
      logging_steps=100,
      save_strategy="no",
      report_to="tensorboard",
  )

  # Create Trainer instance
  trainer = Seq2SeqTrainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=tokenized_dataset["train"],
  )
  model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


  # train model
  trainer.train()  


  # Compute rouge
  path_to_data = "data/eval"   # path to test data
  predictions, references, rogue = compute_rouge(model, tokenizer, path_to_data)

  save_results(dataset_name, model_name, predictions, references, rogue)

  # Save our model & tokenizer results
  print(" Save the model & tokenizer results")
  peft_model_id="results"
  trainer.model.save_pretrained(peft_model_id)
  tokenizer.save_pretrained(peft_model_id)

  print("finished.")



def main(args):

  model_names = [sys.argv[1]] if len(args) > 1 else ['t5', 'bart']
  ds_names = ['hdfs', 'hadoop', 'bgl', 'thunderbird']
  
  for model_name in model_names:
    if model_name == "bart":
      model_name = "bart-large-xsum"

    print(f"model_name: {model_name}")
    # model_name = "t5"
    for dataset_name in ds_names:
      # dataset_name = "BGL"
      print(f"dataset_name:{dataset_name}")
      path_to_csv = f"./data/{dataset_name}_2k.log_structured.csv"
      dataset = load_dataset(dataset_name, path_to_csv)
      train(model_name, dataset_name, dataset)


if __name__ == "__main__":
  main(sys.argv)
