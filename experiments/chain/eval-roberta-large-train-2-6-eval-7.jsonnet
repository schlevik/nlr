local base_name = "train-roberta-large-guaranteed-depths-2-6/";
local run_name = base_name + "eval-depths-7-mi15-ma30";
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    #do_train: true,
    do_eval: true,
    validation_file: "data/eval-guaranteed-d7-s56-mi15-ma30-k100.json",
    train_file: "data/train-guaranteed-dmin2-dmax6-s1337-mi15-ma30-k1000.json",
    overwrite_cache: true,
    max_seq_length: 288,
    per_device_train_batch_size: 56,
    per_device_eval_batch_size: 64,
    #learning_rate: 3e-6,
    #num_train_epochs: 6,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
