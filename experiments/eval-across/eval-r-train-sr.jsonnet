local base_name = "roberta-sr-final";
local run_name = "eval-r" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/eval-e-40-45.json",
    validation_file: "data/eval-no-neg-hard0na200nb2nfmin15nfmax30pr0.2pus0.8puf0.8pue0.8puu0.8pns0.0pno0.0pnp0.5s500sd56.json",
    overwrite_cache: true,
    max_seq_length: 288,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
