import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

folds = 10
enum = 50

pretrained_avaliable = True

validation = False
label = 'valid_loss' if validation else 'train_loss'

fig, ax = plt.subplots()

for k in range(folds):
    if k == 6:
        continue
    data = pd.read_csv(f'record/just_sm_303_50_enum/REG_{enum}_FOLD-{k}_LOSS.csv')
    ax.plot(label, data=data, color='b', alpha=0.5)

    if pretrained_avaliable:
        data = pd.read_csv(f'record/pretrained_50_enum/REG_{enum}_FOLD-{k}_LOSS.csv')
        ax.plot(label, data=data, color='y', alpha=0.5)


ax.set_ylim([-0.05, 1.05])
ax.plot([], color='b',label='RNN')
ax.legend(['RNN', 'SRNN'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
plt.show()