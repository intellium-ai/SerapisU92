# RNN
Code adapted from original repository and paper.

```
└── RNN
    ├── configs.py
    ├── data
    │   ├── Dm.csv - Original labelled dataset.
    │   └── Ds_9.csv
    ├── generated.csv
    ├── generated_scored.csv - Generated molecules with their DV and SA.
    ├── graph_generated.py - Displays generated molecules.
    ├── graph_pred_loss.py
    ├── launcher_of_clm.py - Pretraining generative model.
    ├── launcher_of_sm.py - Fine tuning predictive model.
    ├── models
    │   ├── __init__.py
    │   ├── clm_rnn.py
    │   └── reg_rnn.py
    ├── README.md
    ├── original-README.md - README from original repo.
    ├── requirements.txt
    └── utils
```