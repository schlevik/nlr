local run_name = "roberta-sr-final";
{
    model_name_or_path: "roberta-large",
    output_dir: "./models/"+run_name,
    overwrite_output_dir: true,
    do_train: true,
    do_eval: true,
    validation_file: "data/eval-c-no-neg-CompSubjna200nfmin15nfmax30pu0.8pnc0.0pnp0.5s500sd56.json",
    train_file: "data/train-c-no-neg-CompSubjna200nfmin15nfmax30pu0.8pnc0.0pnp0.5s3750sd1337.json",
    overwrite_cache: true,
    max_seq_length: 432,
    per_device_train_batch_size: 40,
    learning_rate: 4e-6,
    num_train_epochs: 9,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",
    warmup_ratio: 0.1
}
