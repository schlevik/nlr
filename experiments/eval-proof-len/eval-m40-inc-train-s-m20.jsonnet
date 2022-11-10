local base_name = "roberta-s-m20";
local run_name = "eval-m40-inc" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/train-s-m20.json",
    validation_file: "data/unseen-s-m40-inc.json",
    overwrite_cache: true,
    max_seq_length: 320,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
