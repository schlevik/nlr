local base_name = "roberta-r-m20";
local run_name = "eval-c-1-2" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/eval-r-30-40.json",
    validation_file: "data/chain-r-1-2.json",
    overwrite_cache: true,
    max_seq_length: 288,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
