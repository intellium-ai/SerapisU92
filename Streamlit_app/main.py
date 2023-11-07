from RNN.launcher_of_clm import valid_generate
from RNN. launcher_of_sm import score

def generate_scored_mol(number_of_molecules: int):
    valid_generate(number_of_molecules, 0, 'data/Dm.csv', 'ft_pretrained_100k.pth', 'generated.csv', None)
    score('data/Dm.csv', 'generated.csv', 'record/reg_50_pretrained.pth', 'generated_scored.csv', 0, 0, 11)
