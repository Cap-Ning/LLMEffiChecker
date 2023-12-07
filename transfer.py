import os
import torch
import time
from utils import MODEL_NAME_LIST, load_model, translate
from tqdm import tqdm
import numpy as np
import argparse
if not os.path.isdir('transfer_res'):
    os.mkdir('transfer_res')
@torch.no_grad()
def measure_transferbility(model, tokenizer, adv_his, max_length, device):
    metric = {
        'flops': [],
    }
    model = model.to(device)

    org_x = adv_his[0][0]
    input_token = tokenizer(org_x, return_tensors="pt", padding=True).input_ids
    input_token = input_token.to(device)
    org_len = len(input_token[0])

    metric['flops'].append(org_len)
    org_pred = model.generate(
            input_token,
            max_length=max_length,
        )

    metric['flops'].append(len(org_pred[0]))                

    attack_x =  adv_his[3][0]   
    input_token = tokenizer(attack_x, return_tensors="pt", padding=True).input_ids
    input_token = input_token.to(device)    
    attack_pred = model.generate(
            input_token,
            max_length=max_length,
        )
    metric['flops'].append(len(attack_pred[0]))  
    return metric['flops']


def main(model_id, attack_id, beam):
    device = torch.device('cuda')
    max_length = 200
    data_name = MODEL_NAME_LIST[model_id]
    model, tokenizer, _, _, _ = load_model(data_name)

    task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(model_id)
    adv_res = torch.load('effect/' + task_name + '_' + str(beam) + '.adv')
    # adv_res = adv_res[:100]
    save_path = os.path.join('transfer_res', str(attack_id) + '_' + str(model_id) + '.res')
    final_res_data = []
    # ori_flops = np.array([d[0][1] for d in adv_res])
    for adv in tqdm(adv_res):
        metric = measure_transferbility(model, tokenizer, adv, max_length, device)
        final_res_data.append((np.array(metric)).reshape([1, -1]))
        torch.save(final_res_data, save_path)
    print(attack_id, model_id, 'successful')


def post():
    attack_name = 0
    model_id = 15
    data_name = MODEL_NAME_LIST[model_id]
    final_res = []
    save_path = os.path.join('transfer_res', str(attack_name) + '_' + str(model_id) + '.res')
    res = torch.load(save_path)
    res = np.concatenate(res).max(0)
    res = res * 100
    res = np.array([(1) * 1000 + (1)] + list(res))
    final_res.append(res.reshape([1, -1]))
    final_res = np.concatenate(final_res)
    np.savetxt('res/trans_' + attack_name + '.csv', final_res, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure Latency')
    parser.add_argument('--data', default=1, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=0, type=int, help='attack type')
    parser.add_argument('--beam', default=5, type=int, help='beam num')
    args = parser.parse_args()
    main(args.data, args.attack,args.beam)
    # post()
