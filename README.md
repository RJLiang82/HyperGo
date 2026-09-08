
# HyperGo
This is the official implementation of Hypergraph-Based Hyper-Relational Knowledge Representation Learning with Gradient Calibration.


# Requirements and Installation
This project should work fine with the following environments:
- Python 3.8.20 for data preprocessing, training, and evaluation with:
    -  pytorch 2.3.1
    -  torch-scatter 2.1.2
    -  torch-geometric 2.6.1
    -  numpy 1.24.3
- GPU with CUDA 12.4

All experiments are implemented in PyTorch on a single NVIDIA GeForce RTX 3090 Ti.

# Training & Evaluation
Please modify those hyperparameters according to your needs and the characteristics of different datasets.

For JF17K, to train and evaluate on this dataset using default hyperparameters, please run:
```
python -u ./src/run.py --dataset "jf17k" --device "0" --vocab_size 29148 --vocab_file "./data/jf17k/vocab.txt" --train_file "./data/jf17k/train.json" --test_file "./data/jf17k/test.json" --ground_truth_file "./data/jf17k/all.json" --num_workers 1 --num_relations 501 --max_seq_len 11 --max_arity 6 --hidden_dim 256 --global_layers 2 --global_dropout 0.9 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.35 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_deca 0.002 --entity_soft 0.9 --relation_soft 0.9 --hyperedge_dropout 0.85 --epoch 300 --warmup_proportion 0.05
```

For Wikipeople, to train and evaluate on this dataset using default hyperparameters, please run:
```
python -u ./src/run.py --dataset "wikipeople" --device "0" --vocab_size 35005 --vocab_file "./data/wikipeople/vocab.txt" --train_file "./data/wikipeople/train+valid.json" --test_file "./data/wikipeople/test.json" --ground_truth_file "./data/wikipeople/all.json" --num_workers 1 --num_relations 178 --max_seq_len 13 --max_arity 7 --hidden_dim 256 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hyperedge_dropout 0.99 --epoch 300 --warmup_proportion 0.1
```

For WD50K, to train and evaluate on this dataset using default hyperparameters, please run:
```
python -u ./src/run.py --dataset "wd50k" --device "0" --vocab_size 47688 --vocab_file "./data/wd50k/vocab.txt" --train_file "./data/wd50k/train+valid.json" --test_file "./data/wd50k/test.json" --ground_truth_file "./data/wd50k/all.json" --num_workers 1 --num_relations 531 --max_seq_len 19 --max_arity 10 --hidden_dim 256 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 12 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 512 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hyperedge_dropout 0.8 --epoch 300 --warmup_proportion 0.1
```
