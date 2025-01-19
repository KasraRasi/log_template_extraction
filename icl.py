import os
import pandas as pd
import torch
import sys
import warnings
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from rouge_metric import PyRouge
from nltk.translate.bleu_score import corpus_bleu

warnings.filterwarnings("ignore", category=UserWarning, module='nltk.translate.bleu_score')

OUTPUT_DIR = 'out/ICL'
NUM_EXAMPLES = 10

rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)


def gpt2_predict(row):
    _input = row['message']
    while True:
        try:

            # Prepare Prompt
            examples = prep_examples(train)
            base_prompt = "Extract one log template, substitute variable tokens in the log as <*> between" \
                          + " <START> and <END> tags."
            final_prompt = base_prompt + examples + f"\n\nMessage: {_input}\n"
            prompt_token_size = len(tokenizer.encode(final_prompt))
            if prompt_token_size > 920:
                continue
            print(f"{row['ds_name']}_{row['row_num']}, prompt size: {prompt_token_size}")
            prompt_size = len(final_prompt)
            inputs = tokenizer.encode(final_prompt, return_tensors="pt", add_special_tokens=True).cuda()

            # Generate Response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=prompt_token_size + 100,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.01,
                    do_sample=True
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Post-Process the Response and Return the Output
            output = response[prompt_size:]
            if '<START>' in output:
                output = output.split('<START>')[1]
            if '<END>' in output:
                output = output.split('<END>')[0]
            output = output.split('\n')[0].strip()
            return output
        except Exception as e:
            print(e)
            pass


def incoder_predict(row):
    BOS = "<|endoftext|>"
    input = row['message']
    while True:
        try:

            # Prepare Prompt
            examples = prep_examples(train)
            base_prompt = "Extract one log template, substitute variable tokens in the log as <*> between" \
                          + " <START> and <END> tags."
            final_prompt = base_prompt + examples + f"\n\nMessage: {input}\n"
            input_ids = tokenizer.encode(final_prompt, return_tensors="pt", add_special_tokens=True).cuda()
            prompt_token_size = input_ids.flatten().size(0)
            if prompt_token_size > 2048:
                continue
            print(f"{row['ds_name']}_{row['row_num']}, prompt size: {prompt_token_size}")
            prompt_size = len(final_prompt)

            # Generate Response
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_length=prompt_token_size + 100,
                    top_p=0.95,
                    temperature=0.5,
                    do_sample=True,
                )
            response = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
            if response.startswith(BOS):
                response = response[len(BOS):]

            # Post-Process the Response and Return the Output
            output = response[prompt_size:]
            if '<START>' in output:
                output = output.split('<START>')[1]
            if '<END>' in output:
                output = output.split('<END>')[0]
            output = output.split('\n')[0].strip()
            return output
        except Exception as e:
            print(e)
            pass


def bart_predict(row):
    input = row['message']
    try:

        # Prepare Prompt
        base_prompt = "extract template:"
        examples = train.sample(NUM_EXAMPLES).to_dict(orient='records')
        examples_str = ""
        for example in examples:
            examples_str += f"{base_prompt} {example['message']} -> [START] {example['template'].replace('<', '[').replace('>', ']')} [END]\n"
        base_prompt = "Extract one log template, substitute variable tokens in the log as <*> between <START> and <END> tags."
        final_prompt = examples_str + f"{base_prompt} {input} ->"
        tokenized = tokenizer(final_prompt, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        prompt_token_size = input_ids.flatten().size(0)
        print(f"{row['ds_name']}_{row['row_num']}, prompt size: {prompt_token_size}")

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=100,
                attention_mask=attention_mask,
                temperature=0.1,
                do_sample=True,
            )

        # Post-Process the Response and Return the Output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = response.split('\n')[0].replace('[', '<').replace(']', '>')
        if '<START>' in output_text:
            output_text = output_text.split('<START>')[1]
        if '<END>' in output_text:
            output_text = output_text.split('<END>')[0]
        return output_text.split('->')[0].strip()
    except Exception as e:
        print(e)
        pass


def t5_predict(row):
    input = row['message']
    try:

        # Prepare Prompt
        base_prompt = "extract template:"
        examples = train.sample(NUM_EXAMPLES).to_dict(orient='records')
        examples_str = ""
        for example in examples:
            examples_str += f"{base_prompt} {example['message']} -> [START] {example['template'].replace('<', '[').replace('>', ']')} [END]\n"
        final_prompt = examples_str + f"{base_prompt} {input} ->"
        tokenized = tokenizer(final_prompt, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        prompt_token_size = input_ids.flatten().size(0)

        print(f"{row['ds_name']}_{row['row_num']}, prompt size: {prompt_token_size}")

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=100,
                attention_mask=attention_mask,
                temperature=0.1,
                do_sample=True,
            )

        # Post-Process the Response and Return the Output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = response.split('\n')[0].replace('[', '<').replace(']', '>')
        if '<START>' in output_text:
            output_text = output_text.split('<START>')[1]
        if '<END>' in output_text:
            output_text = output_text.split('<END>')[0]
        return output_text.split('->')[0].strip()
    except Exception as e:
        print(e)
        pass


def get_token_count(final_prompt, tokenizer):
    tokens = tokenizer.encode(final_prompt)
    print(f"Number of tokens: {len(tokens)}")


def prep_examples(train):
    examples = train.sample(NUM_EXAMPLES).to_dict(orient='records')
    examples_str = ""
    for example in examples:
        examples_str += f"\n\nMessage: {example['message']}\nTemplate: <START> {example['template']} <END>"
    return examples_str


def run(model_names, ds_names, num_examples):
    for model_name in model_names:
        print(f'Generating predictions for {model_name}...')
        global tokenizer
        global model
        tokenizer = MODEL_SPECIFICATIONS[model_name]['tokenizer']
        model = MODEL_SPECIFICATIONS[model_name]['model'].cuda()
        func = MODEL_SPECIFICATIONS[model_name]['func']

        rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                        rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

        for ds_name in ds_names:
            csv_file_path = f'data/{ds_name}_2k.log_structured.csv'

            struct_log = pd.read_csv(csv_file_path).rename(columns={
                'Content': 'message',
                'EventTemplate': 'template',
            })[['message', 'template']].sample(frac=1)

            global train
            train = struct_log.iloc[:1800]
            test = struct_log.iloc[1800:].copy()
            test['row_num'] = list(range(len(test)))
            test['ds_name'] = ds_name
            test['pred'] = test.apply(func, axis=1)

            # Save Outputs
            test.to_pickle(os.path.join(OUTPUT_DIR, 'predictions', f'{ds_name}_{model_name}_predictions.pkl'))
            scores = rouge.evaluate([p for p in test['pred']], [[p] for p in test['template']])
            print(f'results for model {model_name} on dataset {ds_name}:')
            print(scores)


def evaluate(model_names, ds_names):
    for model_name in model_names:
        print(f'Evaluating {model_name}...')
        res = pd.DataFrame()
        for ds_name in ds_names:
            predictions = pd.read_pickle(
                os.path.join(OUTPUT_DIR, 'predictions', f"{ds_name}_{model_name}_predictions.pkl"))
            predictions['correct'] = predictions['pred'] == predictions['template']
            scores = rouge.evaluate([p for p in predictions['pred']], [[p] for p in predictions['template']])
            reference = [[p.split()] for p in predictions['template']]
            candidate = [p.split() for p in predictions['pred']]
            template_df = predictions.groupby('template').agg({'row_num': 'count', 'correct': 'sum'}) \
                .rename(columns={'row_num': 'count'})
            correctly_identified = len(template_df[template_df['correct'] > 0])
            predicted_templates = len(set(predictions['pred']))
            row = {
                'Dataset': ds_name,
                'Rouge-1': f"{scores['rouge-1']['f'] * 100:.2f}",
                'Rouge-2': f"{scores['rouge-2']['f'] * 100:.2f}",
                'Rouge-L': f"{scores['rouge-l']['f'] * 100:.2f}",
                'BLEU': f"{corpus_bleu(reference, candidate):.2f}",
                'PA': f"{predictions['correct'].sum() / len(predictions):.2f}",
                'PTA': f"{correctly_identified / predicted_templates:.2f}",
                'RTA': f"{correctly_identified / len(template_df):.2f}",
            }
            res = pd.concat([res, pd.DataFrame([row])])
        save_to = os.path.join(OUTPUT_DIR, 'results', f'results_{model_name}.csv')
        res.to_csv(save_to, index=False)
        print(
            f'The evaluations of model {model_name} has been successfully saved to {save_to}.'
        )


def main(args):
    model_names = [sys.argv[1]] if len(args) > 1 else ['gpt-2', 'incoder', 't5', 'bart']
    ds_names = ['hdfs', 'hadoop', 'bgl', 'thunderbird']
    run(model_names, ds_names, num_examples=10)
    evaluate(model_names, ds_names)


if __name__ == '__main__':
    print('Please wait for all models to be downloaded...')
    MODEL_SPECIFICATIONS = {
        'gpt-2': {
            'tokenizer': GPT2Tokenizer.from_pretrained("gpt2-xl"),
            'model': GPT2LMHeadModel.from_pretrained("gpt2-xl"),
            'func': gpt2_predict,
        },
        'incoder': {
            'tokenizer': AutoTokenizer.from_pretrained("facebook/incoder-1B"),
            'model': AutoModelForCausalLM.from_pretrained("facebook/incoder-1B", low_cpu_mem_usage=True).half(),
            'func': incoder_predict,
        },
        't5': {
            'tokenizer': T5Tokenizer.from_pretrained("google/flan-t5-xxl"),
            'model': T5ForConditionalGeneration.from_pretrained("philschmid/flan-t5-base-samsum"),
            'func': t5_predict,
        },
        'bart': {
            'tokenizer': BartTokenizer.from_pretrained('facebook/bart-large'),
            'model': BartForConditionalGeneration.from_pretrained('facebook/bart-large'),
            'func': bart_predict,
        }
    }
    main(sys.argv)
