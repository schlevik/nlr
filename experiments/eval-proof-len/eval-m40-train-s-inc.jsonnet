local base_name = "roberta-s-final";
local run_name = "eval-m40-inc" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/train-s-na200nb2nfmin15nfmax30pr0.0pus0.8puf0.8pue0.8puu0.8pns0.5pno0.5pnp0.5s3750sd1337.json",
    validation_file: "data/hard-eval-s-m40-inc.json",
    overwrite_cache: true,
    max_seq_length: 320,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
