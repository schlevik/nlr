local base_name = "train-chain-r+s";
local run_name = "eval-c7-c10-" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/eval-guaranteed-dmin7-dmax10-mi15-ma30-k100.json",
    validation_file: "data/eval-guaranteed-dmin7-dmax10-mi15-ma30-k100.json",
    overwrite_cache: true,
    max_seq_length: 384,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
