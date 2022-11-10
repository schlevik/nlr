local base_name = "roberta-all";
local run_name = "eval-s-h-" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    overwrite_output_dir: true,
    train_file: "data/eval-s-na200nb2nfmin15nfmax30pr0.0pus0.8puf0.8pue0.8puu0.8pns0.5pno0.5pnp0.5s500sd56.json",
    validation_file: "data/eval-s-hard0na200nb2nfmin15nfmax30pr0.0pus0.8puf0.8pue0.8puu0.8pns0.5pno0.5pnp0.5s500sd56.json",
    overwrite_cache: true,
    max_seq_length: 320,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
