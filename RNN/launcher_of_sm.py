import numpy as np
import time
import torch
import torch.optim as optim

import configs as cfg
from utils import builder, tools, smi_tools, augmentation
from models import reg_rnn
import math

def train_predictor(data_path: str, pretrained_path: str, target_index=1, epochs=100, k=10, SMILE_enumeration_level=100, save_filename: str = 'model'):
    """
    :param data_path:
    :param pretrained_path:
    :param target_index:
    :param epochs:
    :param SMILE_enumeration_level: (0/50/100/200)
    :return:
    """

    records = {'n_train': [], 'n_test': [], 'n_augment': [], 'mae': [], 'rmse': [], 'r2': []}
    time_start = time.time()

    data, column_names = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [x[0] for x in data]
    tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)
    print(f'Tokens: {tokens}')
    print(f'Property name: {column_names[target_index]}')
    lst_prop = [float(x[target_index]) for x in data]


    folds = tools.train_test_split(smiles, lst_prop, k)
    #assert smi_tools.if_each_fold_cover_all_tokens(folds, tokens), f'[ERROR] Not every fold cover all tokens!'

    for i in range(k):
        #print(f"Fold: {i}")
        train_smi, train_y, test_smi, test_y = folds[i]
        norm = tools.ZeroSoreNorm(train_y)
        train_y, test_y = norm.norm(train_y), norm.norm(test_y)

        records['n_train'].append(len(train_smi))
        records['n_test'].append(len(test_smi))

        if SMILE_enumeration_level > 0:
            #print('Start augmentation ...')
            #print(f'Max enumeration times: {SMILE_enumeration_level}.')
            augmented_smi, augmented_y = augmentation.augmentation_by_enum(train_smi, train_y, max_times=SMILE_enumeration_level)
            remain_idx = [idx for idx, smi in enumerate(augmented_smi) if smi_tools.if_oov_exclude(smi, tokens)]
            train_smi = [augmented_smi[idx] for idx in remain_idx]
            train_y = [augmented_y[idx] for idx in remain_idx]
            #print(f'Augmentation finished. {len(augmented_smi)} augmented, {len(train_smi)} accepted.')

        records['n_augment'].append(len(train_smi))
        train_oh = [smi_tools.smiles2tensor(smi, tokens) for smi in train_smi]
        test_oh = [smi_tools.smiles2tensor(smi, tokens) for smi in test_smi]

        loader = builder.reg_packer(train_oh, train_y)


        model = builder.build_reg(len(tokens), pretrained_path, True if pretrained_path else False)
        opt = optim.Adam(model.parameters(), lr=cfg.REG_LR_RATE, weight_decay=0.01)

        #print(f'Dataloader - Num workers: {loader.num_workers}, memory pinned: {loader.pin_memory}, batch size: {loader.batch_size}')


        name_of_m = f'reg_pre_{i}' if pretrained_path else f'reg_non_{i}'
        losses = reg_rnn.train(model=model, optimizer=opt, train_loader=loader, valid_x=test_oh, valid_y=test_y,
                               epochs=epochs, filename=f"{save_filename}-{i}")

        #tools.save_data_to_csv(f'training_records/REG_{SMILE_enumeration_level}_FOLD-{i}_LOSS.csv', losses, ['epoch', 'train_loss', 'valid_loss'])

        y_pred_train = builder.normed_predict(train_oh, model, norm)
        y_train = norm.recovery(np.array(train_y))

        y_pred_test = builder.normed_predict(test_oh, model, norm)
        y_test = norm.recovery(np.array(test_y))

        results = list(zip(train_smi, y_train, y_pred_train, [0] * len(y_train)))
        results += list(zip(test_smi, y_test, y_pred_test, [1] * len(y_test)))
        #tools.save_data_to_csv(f'training_records/REG_{SMILE_enumeration_level}_FOLD-{i}_RESULTS.csv', results, ['smi', 'y', 'y_pred', 'test'])

        #print(f"Train MSE: {tools.mse(y_train, y_pred_train)}, R2: {tools.r_square(y_train, y_pred_train)}")
        #print(f"Test MSE: {tools.mse(y_test, y_pred_test)}, R2: {tools.r_square(y_test, y_pred_test)}")

        records['mae'].append(np.mean(np.abs(y_pred_test - y_test)))
        records['rmse'].append(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
        records['r2'].append(tools.r_square(y_test, y_pred_test))

    rcd = list(np.array(list(records.values())).T)
    tools.save_data_to_csv(f'training_records/{save_filename}_Train_Performance.csv', rcd,
                                ['train', 'test', 'augment', 'mae', 'rmse', 'r2'])
    print('--- Final Results ---')
    print('mae', sum(records['mae'])/k)
    print('rmse', sum(records['rmse'])/k)
    print('r2', sum(records['r2'])/k)

    time_end = time.time()
    print(f'Time: {time_end-time_start}')

def score(train_data_path: str, data_path: str, model_path: str, saving_path: str,
          SMILE_index_1: int, SMILE_index_2: int, target_index: int):

    data, head = tools.load_data_from_csv(train_data_path, with_head=True)
    smiles = [x[SMILE_index_1] for x in data]
    tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)
    #print(tokens, '\n\n')

    list_of_prop = [float(x[target_index]) for x in data]
    norm = tools.ZeroSoreNorm(list_of_prop)
    #print(head[target_index], norm.avg, norm.std)

    data, _ = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [x[SMILE_index_2] for x in data]

    for smi in smiles:
        if not smi_tools.if_oov_exclude(smi, tokens):
            print(f'Excluding smile due to an unknown token. {smi}')

    raw_smiles = [smi for smi in smiles if smi_tools.if_oov_exclude(smi, tokens)]
    smiles = [smi for smi in raw_smiles]

    m = builder.build_reg(len(tokens), model_path, False)
    y_predict = builder.smiles_predict(smiles, tokens, m, norm)

    sa_scores = smi_tools.get_sa(raw_smiles)
    #print(np.mean(sa_scores), np.std(sa_scores))

    data = [[raw_smiles[idx], y_predict[idx], sa_scores[idx]] for idx in range(len(raw_smiles))]
    tools.save_data_to_csv(saving_path, data, head=['smiles', head[target_index], 'SA'])


if __name__ == '__main__':
    #train_predictor('data/Dm.csv', 'ft.pth', target_index=11, epochs=100, k=10, SMILE_enumeration_level=50)
    score('data/Dm.csv', 'three_isomers.csv', 'record/reg_50_pretrained_density.pth', 'three_isomers_scored.csv', 0, 0, 7)
    #train_predictor('data/Dm.csv', 'streamlit_utils/models/ft_pretrained_100k.pth', target_index=7, epochs=100, k=10, SMILE_enumeration_level=50)
