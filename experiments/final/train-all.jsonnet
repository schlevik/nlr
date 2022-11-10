local run_name = "roberta-all-longer";
{
    model_name_or_path: "roberta-large",
    output_dir: "./models/"+run_name,
    overwrite_output_dir: true,
    do_train: true,
    do_eval: true,
    validation_file: "data/eval-r-filtered.json",
    train_file: "data/train-all-filtered.json",
    overwrite_cache: true,
    max_seq_length: 432,
    per_device_train_batch_size: 56,
    learning_rate: 4e-6,
    num_train_epochs: 12,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",
    warmup_ratio: 0.1
}
