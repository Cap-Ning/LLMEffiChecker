import datetime
import os
import torch
import argparse
import numpy as np

from utils import *


if not os.path.isdir('adv'):
    os.mkdir('adv')

MODEL_NAME_LIST = [
    'Helsinki-en-zh',
    'facebook-wmt19',
    'T5-small',
    'allenai-wmt16',
    'DDDSSS',
    'unicamp',
    'flan-t5-small',
]
# MODEL_NAME_LIST = [
#     'LaMini-GPT',
# ]

for data_name in MODEL_NAME_LIST:
    for attack_type in [0, 1, 6, 7, 8, 9]:
        succ_num=0
        device = torch.device('cuda')
        model, tokenizer, space_token, dataset, src_lang, tgt_lang = load_model_dataset(data_name)
        print('load model %s successful' % data_name)
        config = {
            'num_beams': model.config.num_beams,
            'num_beam_groups': model.config.num_beam_groups,
            'max_per': 3,
            'max_len': 100,
            'src': src_lang,
            'tgt': tgt_lang
        }
        if attack_type == 0:
            attack = CharacterAttack(model, tokenizer, space_token, device, config)
            attack_name = 'C'
        elif attack_type == 1:
            attack = WordAttack(model, tokenizer, space_token, device, config)
            attack_name = 'W'
        elif attack_type == 6:
            attack = StructureAttack(model, tokenizer, space_token, device, config)
            attack_name = 'S'
        elif attack_type == 7:
            attack = Black_box_CharacterAttack(model, tokenizer, space_token, device, config)
            attack_name = 'BC'
        elif attack_type == 8:
            attack = Black_box_StructureAttack(model, tokenizer, space_token, device, config)
            attack_name = 'BW'
        elif attack_type == 9:
            attack = Black_box_WordAttack(model, tokenizer, space_token, device, config)
            attack_name = 'BS'        
        else:
            raise NotImplementedError
        results = []
        for i, src_text in enumerate(dataset):
            if succ_num>=20:
                break
            if i == 0:
                continue
            if i >= 100:
                break
            src_text = src_text.replace('\n', '')
            if attack_type in [7,8,9]:
                is_success, adv_his = attack.run_black_attack([src_text])
            else:
                is_success, adv_his = attack.run_attack([src_text])
            if is_success:
                succ_num=succ_num+1
                overhead = np.array([r[-1] for r in adv_his]).reshape([1, -1])
                results.append(overhead)
        print(np.concatenate(results)[:,1:].mean(0))




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Transformer')
#     parser.add_argument('--data', default=2, type=int, help='experiment subjects')
#     parser.add_argument('--attack', default=1, type=int, help='attack type')
#     args = parser.parse_args()
#     main()
