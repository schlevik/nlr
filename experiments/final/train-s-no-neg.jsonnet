local run_name = "roberta-s-no-neg-final";
{
    model_name_or_path: "roberta-large",
    output_dir: "./models/"+run_name,
    overwrite_output_dir: true,
    do_train: true,
    do_eval: true,
    train_file: "data/train-s-no-neg-hard0na200nb2nfmin15nfmax30pr0.0pus0.8puf0.8pue0.8puu0.8pns0.0pno0.0pnp0.5s3750sd1337.json",
    validation_file: "data/eval-s-no-neg-hard0na200nb2nfmin15nfmax30pr0.0pus0.8puf0.8pue0.8puu0.8pns0.0pno0.0pnp0.5s500sd56.json",
    overwrite_cache: true,
    max_seq_length: 256,
    per_device_train_batch_size: 56,
    learning_rate: 4e-6,
    num_train_epochs: 6,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",
    warmup_ratio: 0.1
}
