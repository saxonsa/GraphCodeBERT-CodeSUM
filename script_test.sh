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
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
epochs=20 
pretrained_model=microsoft/graphcodebert-base #Roberta: roberta-base

python run.py --do_test --model_type roberta --model_name_or_path microsoft/graphcodebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size --lang $lang
