local base_name = "roberta-r-e-m20";
local run_name = "eval-c-3" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/eval-e-30-40.json",
    validation_file: "data/chain-r-e-3.json",
    overwrite_cache: true,
    max_seq_length: 404,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
