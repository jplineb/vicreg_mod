# Snakefile for automated evaluation of models
configfile: "config/config.yaml"

# Define the target rule that generates all outputs
rule all:
    input:
        expand(
            "{exp_dir}/{task_ds}/{model_type}_{weights}/results.txt",
            exp_dir=config["base_params"]["exp_dir"],
            task_ds=config["task_datasets"],
            model_type=config["pretrained_models"].keys(),
            weights=config["weight_strategies"]
        )

# Rule to run the evaluation script
rule run_evaluate:
    output:
        "{exp_dir}/{task_ds}/{model_type}_{weights}/results.txt"
    params:
        task_ds = lambda wildcards: wildcards.task_ds,
        model_type = lambda wildcards: wildcards.model_type,
        weights = lambda wildcards: wildcards.weights,
        seed = config["base_params"]["seed"],
        pretrained_how = lambda wildcards: config["pretrained_models"][wildcards.model_type]["pretrained_how"],
        pretrained_dataset = lambda wildcards: config["pretrained_models"][wildcards.model_type]["pretrained_dataset"],
        pretrained_path = lambda wildcards: config["pretrained_models"][wildcards.model_type]["pretrained_path"],
        exp_subdir = lambda wildcards: f"{wildcards.exp_dir}/{wildcards.task_ds}/{wildcards.model_type}_{wildcards.weights}",
        epochs = config["base_params"]["epochs"],
        lr_backbone = config["base_params"]["lr_backbone"],
        lr_head = config["base_params"]["lr_head"],
        batch_size = config["base_params"]["batch_size"],
        weight_decay = config["base_params"]["weight_decay"],
        workers = config["base_params"]["workers"],
        warmup_epochs = config["base_params"]["warmup_epochs"]
    resources:
        # These resources will be used if not overridden by --default-resources
        # slurm_partition = "wficai",
        # slurm_account = "wficai",
        runtime = 3600,
        cpus_per_task = 32,
        mem_mb = 128000,
        slurm_extra = "'--gpus=v100s:1'"
    log:
        "logs/{exp_dir}/{task_ds}/{model_type}_{weights}/slurm.log"
    shell:
        """
        # Make sure the output directory exists
        mkdir -p {params.exp_subdir}
        mkdir -p $(dirname {log})
        
        # Set up environment (adjust as needed for your system)
        module load miniforge3/24.3.0-0
        module load cuda/12.3

        # Activate the conda environment with verbose output
        source activate disres
        
        # Run the evaluation script with seed parameter
        python evaluate_new.py \
            --task_ds {params.task_ds} \
            --pretrained_path {params.pretrained_path} \
            --exp-dir {params.exp_subdir} \
            --pretrained-how {params.pretrained_how} \
            --pretrained-dataset {params.pretrained_dataset} \
            --epochs {params.epochs} \
            --weights {params.weights} \
            --lr-backbone {params.lr_backbone} \
            --lr-head {params.lr_head} \
            --batch-size {params.batch_size} \
            --weight-decay {params.weight_decay} \
            --workers {params.workers} \
            --warmup-epochs {params.warmup_epochs} \
            --seed {params.seed} \
            > {output} 2>&1
        """

# rule check_envs:
#     output:
#         "logs/check_envs.txt"
#     resources:
#             # These resources will be used if not overridden by --default-resources
#             slurm_partition = "work1",
#             runtime = 5,  # 6 hours in minutes
#             cpus_per_task = 16,
#             mem_mb = 64000,  # 128GB
#             # slurm_extra = "'--gpus=v100:1'"
#     shell:
#         """
#         mamba info --envs
#         source activate disres

#         python test_snakemake.py  > {output} 2>&1
#         """
