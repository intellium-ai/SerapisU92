import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('generated_scored.csv')

highlight = df[df.D > 8.927][df.SA < 4.03]

plt.scatter(df['D'], df['SA'], marker='.', alpha=0.8)
plt.scatter(highlight['D'], highlight['SA'], marker='.', alpha=1, c='gold')
plt.xlabel('Detonataion Velocity')
plt.ylabel('SA Score')
plt.show()

pd.set_option('display.max_colwidth', None)
print(highlight)