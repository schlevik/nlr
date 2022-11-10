local base_name = "train-roberta-large-dmin2-dmax6-mi15-ma30-realised-neg-subj-warmup";
local run_name = base_name + "eval-depths-9-mi15-ma30-realised-neg-subj";
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    #do_train: true,
    do_eval: true,
    validation_file: "data/eval-guaranteed-d9-realised-s286-mi15-ma30-k100.json",
    train_file: "data/train-guaranteed-dmin2-dmax6-realised-s1337-mi15-ma30-k1000.json",
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
