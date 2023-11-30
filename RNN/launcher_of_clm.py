import torch
import torch.optim as optim

import configs as cfg
from utils import builder, smi_tools, tools
from models import clm_rnn
import os
import timeit


def train_clm(dataset_path: str, SMILE_index: int, model_name='clm', epochs=100, model_path=None):
    print(f"Using device: {cfg.DEVICE}")
    # processing smiles
    if os.path.exists(dataset_path + '.pt'):
        print(f"Loading tokenised {dataset_path + '.pt'}")
        data = torch.load(dataset_path + '.pt')
    else:
        print(f"Tokenising {dataset_path}")
        data, _ = tools.load_data_from_csv(dataset_path, with_head=True)
        smiles = [cfg.BOS + x[SMILE_index] + cfg.EOS for x in data]
        tokens = smi_tools.gather_tokens(smiles, single_split=cfg.SINGLE_TOKENIZE)
        print(f'Tokens: {tokens}')
        print(f'There are {len(smiles)} SMILES strings in data.')
        smiles = [smi for smi in smiles if smi_tools.if_oov_exclude(smi, tokens, single_split=cfg.SINGLE_TOKENIZE)]
        print(f'There are {len(smiles)} SMILES strings after checking tokens.')

        data = builder.clm_packer(smiles, tokens)

        torch.save(data, dataset_path + '.pt')

    #data = torch.utils.data.Subset(data, torch.arange(100_000))
    loader = torch.utils.data.DataLoader(data, batch_size=cfg.CLM_BATCH_SIZE, shuffle=True, num_workers=cfg.REG_NUM_WORKERS, pin_memory=cfg.REG_PIN_MEM)

    num_tokens = next(iter(loader))[0].shape[2]

    model = builder.build_clm(num_tokens, model_path)
    optimizer = optim.Adam(model.parameters(), lr=cfg.CLM_LR_RATE)


    print(f'Dataloader - Num workers: {loader.num_workers}, memory pinned: {loader.pin_memory}, batch size: {loader.batch_size}, Number of data points: {len(data):_}')
    starttime = timeit.default_timer()

    records = clm_rnn.train(model=model, optimizer=optimizer, data_loader=loader,
                            epochs=epochs, name=model_name)
    
    print("Time elapsed:", timeit.default_timer() - starttime)



def generate(n: int, idx: int, data_path: str, model_path: str, saving_path: str) -> list:
    # processing smiles
    data, head = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [x[idx] for x in data]
    raw_smiles = [smi_tools.to_canonical_smi(smi) for smi in smiles]
    raw_smiles = [smi for smi in raw_smiles if smi is not None]

    # initialize clm
    tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)
    m = builder.build_clm(len(tokens), model_path)

    # sampling
    print('Sampling ...')
    novel_smiles, record = builder.generate(n, m, raw_smiles, tokens)
    print('Sampling Finished !')
    print(f'Sample:{n}, Valid:{len(record[1])}, Unique:{len(record[2])}, Novel:{len(record[3])}')
    tools.save_data_to_csv(saving_path, [[smi] for smi in novel_smiles], ['smiles'])

    return novel_smiles


def valid_generate(valid: int, idx: int, data_path: str, model_path: str, saving_path: str, tokens: list) -> list:
    # processing smiles
    data, head = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [x[idx] for x in data]
    raw_smiles = [smi_tools.to_canonical_smi(smi) for smi in smiles]
    raw_smiles = [smi for smi in raw_smiles if smi is not None]

    # initialize clm
    if not tokens:
        tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)
    m = builder.build_clm(len(tokens), model_path)

    # sampling
    valid_smiles = []
    while valid > 0:
        if valid % 100 == 0:
            print(f'\nGenerated {20000 - valid} SMILES\n')
        generate_smiles = builder.sampling(m, tokens, n=1)
        generate_smiles = smi_tools.to_canonical_smi(generate_smiles[0])
        if generate_smiles:
            valid -= 1
            if generate_smiles in valid_smiles:
                print(True)
            valid_smiles.append(generate_smiles)
    print(len(valid_smiles))
    unique_smiles = list(set(valid_smiles))
    novel_smiles = [smi for smi in unique_smiles if smi not in raw_smiles]
    tools.save_data_to_csv(saving_path, [[smi] for smi in novel_smiles], ['smiles'])
    print(f'Valid: {len(valid_smiles)}, Unique: {len(unique_smiles)}, Novel: {len(novel_smiles)}')

    return novel_smiles

def generate_novel_smiles(num_novel: int, idx: int,data_path: str, model_path: str, saving_path: str, tokens: list = None) -> list:
   # processing smiles
    data, head = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [x[idx] for x in data]
    raw_smiles = [smi_tools.to_canonical_smi(smi) for smi in smiles]
    raw_smiles = [smi for smi in raw_smiles if smi is not None]
    novel_smiles = []
    # initialize clm
    if not tokens:
        tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)
    m = builder.build_clm(len(tokens), model_path)
    valid_counter = 0
    while len(novel_smiles) < num_novel:
        if valid_counter % 100 == 0:
            print(f'\nGenerated {valid_counter} Valid SMILES\n')
            print(f'\nGenerated {len(novel_smiles)} Novel SMILES\n')
        generate_smiles = builder.sampling(m, tokens, n=1)
        generate_smiles = smi_tools.to_canonical_smi(generate_smiles[0])
        if generate_smiles:
            valid_counter += 1
            if generate_smiles not in novel_smiles and generate_smiles not in raw_smiles:
                novel_smiles.append(generate_smiles)
    # Save novel smiles to CSV
    tools.save_data_to_csv(saving_path, [[smi] for smi in novel_smiles], ['smiles'])

    print(f'Generated {len(novel_smiles)} novel SMILES')
    return novel_smiles


if __name__ == '__main__':
    #train_clm(dataset_path='data/Dm.csv', SMILE_index=0, model_name='not_pt_303', epochs=100)
    #valid_generate(1000, 0, 'data/Dm.csv', 'ft.pth', 'generated.csv', None)
    generate_novel_smiles(1000, 0, 'data/Dm.csv', 'ft.pth', 'generated.csv', None)
    #train_clm(dataset_path='data/Ds_9.csv', SMILE_index=0, model_name='pt', epochs=100)
    #train_clm(dataset_path='data/Dm.csv', SMILE_index=0, model_name='ft', epochs=100, model_path="pt.pth")
    