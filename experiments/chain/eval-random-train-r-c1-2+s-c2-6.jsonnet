local run_name = "train-chain-r+s";
{
    model_name_or_path: "./models/"+run_name,
    output_dir: "./models/"+"eval-random-r"+run_name,
    do_train: true,
    do_eval: true,
    validation_file: "data/eval-no-neg-hard0na200nb2nfmin15nfmax30pr0.2pus0.8puf0.8pue0.8puu0.8pns0.0pno0.0pnp0.5s500sd56.json",
    train_file: "data/eval-no-neg-hard0na200nb2nfmin15nfmax30pr0.2pus0.8puf0.8pue0.8puu0.8pns0.0pno0.0pnp0.5s500sd56.json",
    overwrite_cache: true,
    max_seq_length: 288,
    per_device_train_batch_size: 56,
    learning_rate: 3e-6,
    num_train_epochs: 9,
    run_name: "eval-random-r"+run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",
    warmup_ratio: 0.1
}
