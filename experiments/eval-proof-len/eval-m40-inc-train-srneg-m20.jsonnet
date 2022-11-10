local base_name = "roberta-srneg-m20";
local run_name = "eval-m40-inc" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    do_train: false,
    train_file: "data/train-srneg-m20.json",
    validation_file: "data/hard-eval-unseen-srneg-m40-inc.json",
    overwrite_cache: true,
    max_seq_length: 432,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
