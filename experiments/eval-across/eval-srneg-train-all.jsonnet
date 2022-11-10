local base_name = "roberta-all";
local run_name = "eval-srneg-" + base_name;
{
    model_name_or_path: "./models/"+base_name,
    output_dir: "./models/"+run_name,
    do_eval: true,
    train_file: "data/eval-CompSubjna200nfmin15nfmax30pu0.8pnc0.5pnp0.5s500sd56.json",
    validation_file: "data/eval-CompSubjna200nfmin15nfmax30pu0.8pnc0.5pnp0.5s500sd56.json",
    overwrite_cache: true,
    max_seq_length: 432,
    per_device_eval_batch_size: 32,
    run_name: run_name,
    evaluation_strategy: "epoch",
    save_strategy: "no",

}
