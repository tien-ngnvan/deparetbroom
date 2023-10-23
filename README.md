# deparetbroom
Processing text for pre-train and finetune **De**nse **pa**ssage **ret**rieval 

## 1. Get mMARCO dataset
Download data from [huggingface](https://huggingface.co/datasets/unicamp-dl/mmarco) or `bash scripts/get_mmarco.sh`

## 2. Filter query train file
First, we need to filter samples out of `{language}_queries.train.tsv` (only for train datasets)
```python
python filter.py
```
## 3. Create dataset training and test
then,
```python
python create_train_dataset.py 
python create_dev_dataset.py
``` 

## 4. Split datasets into train / validation / test
```python
python create_triplets_dataset.py
```
