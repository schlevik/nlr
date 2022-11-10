local base_name = "roberta-sr-m20";
local run_name = "eval-inc-easy" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/train-sr-m20.json",
    validation_file: "data/eval-sr-inc-easy.json",
    overwrite_cache: true,
    max_seq_length: 432,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
