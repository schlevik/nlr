local run_name = "train-chain-r-1-2";
{
    model_name_or_path: "roberta-large",
    output_dir: "./models/"+run_name,
    do_train: true,
    do_eval: true,
    validation_file: "data/chain-r-1-2.json",
    train_file: "data/train-chain-r-1-2.json",
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
