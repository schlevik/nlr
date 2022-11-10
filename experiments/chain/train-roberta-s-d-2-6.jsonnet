local run_name = "robera-s-final";
{
    model_name_or_path: "roberta-large",
    output_dir: "./models/"+run_name,
    do_train: true,
    do_eval: true,
    validation_file: "data/eval-guaranteed-dmin2-dmax6-realised-s56-mi15-ma30-k100.json",
    train_file: "data/train-guaranteed-dmin2-dmax6-realised-s1337-mi15-ma30-k1000.json",
    overwrite_cache: true,
    max_seq_length: 288,
    per_device_train_batch_size: 56,
    learning_rate: 3e-6,
    num_train_epochs: 9,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",
    warmup_ratio: 0.1
}
