import os
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
import json

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./workdir/huggingface/")
from src import *

BEAM_LIST = [4, 5, 1, 5, 4, 4, 1, 1, 1, 1]

MODEL_NAME_LIST = [
    'Helsinki-en-de',
    'facebook-wmt19',
    'T5-small',
    'allenai-wmt16',
 
    'opus-mt-de-en',

    'DDDSSS',
    'unicamp',
    'LaMini-GPT',
    'flan-t5-small',
    'Codegen',

]
MODEL_WEIGHT = 'model_weight'
ATTACKLIST = [
    CharacterAttack,
    WordAttack,

    Seq2SickAttack,
    NoisyAttack,

    SITAttack,
    TransRepairAttack,

    StructureAttack,
    Black_box_CharacterAttack,
    Black_box_StructureAttack,
    Black_box_WordAttack
]


if not os.path.isdir('res'):
    os.mkdir('res')
if not os.path.isdir(MODEL_WEIGHT):
    os.mkdir(MODEL_WEIGHT)

def load_model(model_name):

    if model_name == 'T5-small':
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        space_token = '▁'
        src_lang, tgt_lang = 'en', 'de'



    elif model_name == 'Helsinki-en-de':
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        space_token = '▁'
        src_lang, tgt_lang = 'en', 'de'

    elif model_name == 'facebook-wmt19':
        tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-de")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-de")
        space_token = '</w>'
        src_lang, tgt_lang = 'en', 'de'

    elif model_name == 'opus-mt-de-en':
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        space_token = '▁'
        src_lang, tgt_lang = 'de', 'en'

    elif model_name == 'allenai-wmt16':
        tokenizer = AutoTokenizer.from_pretrained("allenai/wmt16-en-de-dist-12-1")
        model = AutoModelForSeq2SeqLM.from_pretrained("allenai/wmt16-en-de-dist-12-1")
        space_token = '</w>'
        src_lang, tgt_lang = 'en', 'de'



    elif model_name == 'facebook-wmt19':
        tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-de")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-de",torch_dtype=torch.float16)
        space_token = '</w>'
        src_lang, tgt_lang = 'en', 'de'

    elif model_name == 'DDDSSS':
        tokenizer = AutoTokenizer.from_pretrained("DDDSSS/translation_en-zh")
        model = AutoModelForSeq2SeqLM.from_pretrained("DDDSSS/translation_en-zh",torch_dtype=torch.float16)
        space_token = '▁'
        src_lang, tgt_lang = 'en', 'zh'



    elif model_name == 'unicamp':
        tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-en-pt-t5")
        model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-en-pt-t5",torch_dtype=torch.float16)
        space_token = '▁'
        src_lang, tgt_lang = 'en', 'pt'



    elif model_name == 'LaMini-GPT':
        tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-GPT-124M",padding_side='left')
        model = AutoModelForCausalLM.from_pretrained("MBZUAI/LaMini-GPT-124M")
        # model = AutoModelForCausalLM.from_pretrained("MBZUAI/LaMini-GPT-124M",output_hidden_states=True)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        space_token = 'Ġ'
        src_lang, tgt_lang = 'en', 'en'
    elif model_name == 'flan-t5-small':
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        space_token = '▁'
        src_lang, tgt_lang = 'en', 'en'
    elif model_name == 'Codegen':
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono",padding_side='left')
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        space_token = 'Ġ'
        src_lang, tgt_lang = 'en', 'en'
    else:
        raise NotImplementedError
    return model, tokenizer, space_token, src_lang, tgt_lang


def my_load_dataset(model_name):
    if model_name == 'Helsinki-en-zh':
        with open('./data/Helsinki-en-zh.txt', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'facebook-wmt19':
        # with open('./data/rapid2019.txt', 'r') as f:
        #     data = f.readlines()
        #     return data
        with open('./data/translation2019zh/valid.en', 'r', encoding='utf-8') as f:
            data = f.readlines()
            return data
    elif model_name == 'T5-small':
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'opus-mt-de-en':
        # with open('./data/wmt14.de', 'r') as f:
        #     data = f.readlines()
        #     return data
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'allenai-wmt16':
        # with open('./data/wmt14_valid.en', 'r') as f:
        #     data = f.readlines()
        #     return data
        with open('./data/translation2019zh/valid.en', 'r', encoding='utf-8') as f:
            data = f.readlines()
            return data

    elif model_name == 'DDDSSS':
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'unicamp':
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'gpt2':
        with open('./data/translation2019zh/valid.en', 'r', encoding='utf-8') as f:
            data = f.readlines()
            return data
    elif model_name == 'llama-3b':
        with open('./data/translation2019zh/valid.en', 'r', encoding='utf-8') as f:
            data = f.readlines()
            return data
    elif model_name == 'Llama-2-7b-hf':
        with open('./data/translation2019zh/valid.en', 'r', encoding='utf-8') as f:
            data = f.readlines()
            return data
    elif model_name == 'open_llama_3b_v2':
        with open('./data/translation2019zh/valid.en', 'r', encoding='utf-8') as f:
            data = f.readlines()
            return data
            

    elif model_name == 'LaMini-GPT':
        dataset = load_dataset("hellaswag")
        data = dataset["validation"]['ctx'][:2000]
        return data
    
    elif model_name == 'flan-t5-small':
        dataset = load_dataset("hellaswag")
        data = dataset["validation"]['ctx'][:2000]
        return data
    elif model_name == 'Codegen':
        with open('data/mbpp.json', 'r') as file:
            data = json.load(file)
        prompt = []
        for text in data:
            prompt.append(text['prompt'])
        return prompt
    else:
        raise NotImplementedError


def load_model_dataset(model_name):
    model, tokenizer, space_token, src_lang, tgt_lang = load_model(model_name)
    dataset = my_load_dataset(model_name)
    return model, tokenizer, space_token, dataset, src_lang, tgt_lang


if __name__ == '__main__':
    m = load_model('Codegen')
    print()
