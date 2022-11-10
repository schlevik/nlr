local base_name = "roberta-srneg-final";
local run_name = "eval-inc" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    do_train: false,
    train_file: "data/train-CompSubjna200nfmin15nfmax30pu0.8pnc0.5pnp0.5s3750sd1337.json",
    validation_file: "data/eval-srneg-inc.json",
    overwrite_cache: true,
    max_seq_length: 432,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
