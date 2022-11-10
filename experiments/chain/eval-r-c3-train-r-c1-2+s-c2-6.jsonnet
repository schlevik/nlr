local base_name = "train-chain-r+s";
local run_name = "eval-c3-" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/r+s.json",
    validation_file: "data/chain-r-3.json",
    overwrite_cache: true,
    max_seq_length: 384,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
