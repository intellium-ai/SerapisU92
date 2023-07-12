# RNNMGM

1. data preprocess

   you can use the dataset int data/

   or using  utils/augmentation.py for data preprocess

2. train

   generation model(launcher_of_clm.py)

```
1. pretain:
	train_clm(data_path='data/Ds_9.csv', SMILE_index=0, model_name='pt', epochs=30)
2. fine-tune:
	train_clm(data_path='data/Dm.csv', SMILE_index=0, model_name='tl', epochs=30)

```

​	predictive model(launcher_of_sm.py)

```
train_predictor('data/Dm.csv', pretrain_path, target_index=11, epochs=100, k=10, SMILE_enumeration_level=100)
note: pretrain_path should be the path of model parameters saved in the training process of pretrained generation model  
```

3 . generation 

```
run the generate() or valid_generate() in launcher_of_clm.py

```

4.Score

```
run the score() in launcher_of_sm.py
```

