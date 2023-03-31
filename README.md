### Download data and preprocess

```shell
unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0

### Fine-tune

To fine-tune encoder-decoder on the dataset

```shell
lang=ruby #programming language
lr=5e-5
batch_size=16
beam_size=10
source_length=256
target_length=128
data_dir=dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=20 
pretrained_model=microsoft/graphcodebert-base #Roberta: roberta-base

python run.py --lang $lang --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs
```

### Test
```shell
lang=ruby #programming language
lr=5e-5
batch_size=16
beam_size=10
source_length=256
target_length=128
data_dir=dataset
output_dir=model/$lang
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
epochs=20 
pretrained_model=microsoft/graphcodebert-base #Roberta: roberta-base

python run.py --do_test --model_type roberta --model_name_or_path microsoft/graphcodebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size --lang $lang
```