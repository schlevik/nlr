local base_name = "roberta-r-final";
local run_name = "inoculate-on-c2" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    overwrite_output_dir: true,
    do_eval: true,
    do_train: true,
    inoculate_eval: "data/chain-r-3.json",
    train_file: "data/train-chain-r-1-2.json",
    validation_file: "data/eval-no-neg-hard0na200nb2nfmin15nfmax30pr0.2pus0.8puf0.8pue0.8puu0.8pns0.0pno0.0pnp0.5s500sd56.json",
    overwrite_cache: true,
    max_seq_length: 384,
    per_device_train_batch_size: 56,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",
    hp_sweep: true,
    max_train_samples: 1000,
    ignore_data_skip: true
}
