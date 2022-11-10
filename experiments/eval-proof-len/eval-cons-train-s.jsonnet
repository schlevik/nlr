local base_name = "roberta-s-final";
local run_name = "eval-cons" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/dummy-data.json",
    validation_file: "data/s-cons.json",
    overwrite_cache: true,
    max_seq_length: 320,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
