# Amazon ML hackathon

## Setting Up

**Install requirements for project**

```shell
pip3 install -r requirements.txt
python3 setup_nltk.py
```

**Download & setup dataset**

```shell
mkdir dataset
wget https://huggingface.co/datasets/vasudevgupta/amazon-ml-hack/resolve/main/test-v2.csv -P dataset
wget https://huggingface.co/datasets/vasudevgupta/amazon-ml-hack/resolve/main/train-v2.csv -P dataset
```

**Initiate model training**

```shell
# switch to code directory
cd src

# for training model (supported on single/multi TPUs/GPUs/CPU)
python3 train.py
```

**Making submission**

```shell
# run this from src directory
python3 make_submission.py

# Note: you will have to setup path of trained model in `MODEL_ID` variable inside script
```
