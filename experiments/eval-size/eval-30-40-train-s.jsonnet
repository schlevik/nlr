local base_name = "roberta-s-final";
local run_name = "eval-30-40" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    do_train: false,
    train_file: "data/eval-s-30-40.json",
    validation_file: "data/eval-s-30-40.json",
    overwrite_cache: true,
    max_seq_length: 432,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
