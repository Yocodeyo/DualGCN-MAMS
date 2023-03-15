# DualGCN-MAMS
The repository consists of implementation of DualGCN-ABSA (https://github.com/CCChenhao997/DualGCN-ABSA) on MAMS dataset which is not explored in the original paper.
## Preparation
1. Download and unzip GloVe vectors(glove.840B.300d.zip) from https://nlp.stanford.edu/projects/glove/ and put it into DualGCN/glove directory.

2. Download the best model best_parser.pt of LAL-Parser and put it into LAL-Parser/best_model directory.



## Execution
1. Prepare vocabulary: ```sh DualGCN/build_vocab.sh```

2. Prepare features from MAMS dataset: Run Prepare_Features.ipynb

3. Train and Evaluate: ```CUDA_VISIBLE_DEVICES=0 python3 ./DualGCN/train.py```
