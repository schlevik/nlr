local base_name = "roberta-srneg-final";
local run_name = "eval-30-37" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    do_train: false,
    train_file: "data/eval-srneg-30-37.json",
    validation_file: "data/eval-srneg-30-37.json",
    overwrite_cache: true,
    max_seq_length: 512,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
