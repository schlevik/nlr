local base_name = "roberta-r-e-final";
local run_name = "eval-40-45454545se_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/eval-e-40-45.json",
    validation_file: "data/eval-e-40-45.json",
    overwrite_cache: true,
    max_seq_length: 512,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
