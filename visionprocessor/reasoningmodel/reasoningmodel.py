#! /usr/bin/python
# requisites:
#! pip install azure-core azure-ai-ml rich huggingface_hub
#! curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
#! export HF_TOKEN="hf_xxxxxxxxx"

from aml_setup import setup
ml_clent, drone_mcqa_data, model, compute, environment = setup()
"""
# Sample multiple choice
Which direction has more population density?
A. North
B. East
C. West
D. South

The Ideal reasoning model response:

<think>
Start by rotating the current image to align with the true North relative to the drone camera. Find people in any of the four quadrants. Select the direction from the quadrant that improves the probability the most. If there are no winners among the quadrants, select the direction based on the direction of travel.
</think>
The presence of habitats such as building or outgoing traffic indicates the direction where population density is most.
Final Answer: C.
"""
trainer = GRPOTrainer(
     model=current_policy,
     reward_funcs=reward_function,
     train_dataset=dataset[script_args.dataset_train_split],
     args=training_args,
     peft_config=get_peft_config(model_args),
     processing_class=tokenizer,
     eval_dataset=(
       dataset[script_args.dataset_test_split]
       if training_args.eval_strategy != "no"
       else None
     ),
     callbacks=[save_mlflow_callback],
)
trainer.train()

def format_reward(completions, **kwargs):
    """
    This function determines whether the predicted answer is in the correct format.
    It checks if the reasoning process is enclosed within <think> and </think> tags,
    while the final answer is enclosed within <answer> and </answer> tags.
    Args:
        completions (list): List of model predictions.
    Returns:
        list: List of rewards (1.0 for correct format, 0.0 for incorrect format)
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
       re.match(pattern, content, re.DOTALL | re.MULTILINE)
       for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]

"""
# The following are configurations to
# run vllm to generate samples
# there are two ways to do this: server and collocate
# collocate mode runs sampler and trainer on same GPU.
use_vllm: True
vllm_mode: "collocate"
vllm_gpu_memory_utilization: 0.25
vllm_tensor_parallel_size: 4
reward_funcs:
- accuracy
- format
# reward = 0.8*accuracy + 0.2*format
"""

from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model
)
from azure.ai.ml.constants import AssetTypes
# the following is a command job that takes grpo config, deepspeed config, dataset and the mdoel parameters and kicks off a distributed job.
command_str = """python train.py \
    --config grpo_trainer_config.yaml \
    --model_name_or_path ${{inputs.model_dir}} \
    --dataset_name ${{inputs.dataset}} \
    --output_dir ${{oututs.checkpoint_folder}} \
    --final_model_save_path ${{outputs.mlflow_model_folder}} \
    --deepspeeed deepspeed_stage3_zero_config.json \
    --mlflow_task_type "chat-completion" """
command_str += f'--base_model_name "model.name"'
job_input = {
     "model_dir": Input(
          path=model.path,
          type=AssetTypes.CUSTOM_MODEL,
     ),
     "mlflow_model_folder": Output(
          type=AssetTypes.CUSTOM_MODEL,
          mode="rw_mount",
     ),
     "checkpoint_folder": Output(
          type=AssetTypes.URI_FOLDER,
          mode="rw_mount",
     )
} # notice checkpoints are saved in a separate folder.
job = command(
    code="./src"
    inputs=job_input,
    command=command_str,
    environment=environment,
    compute=compute.name,
    instance_count=2,
    outputs=job_output,
    distribution={
       "type": "PyTorch",
        "process_count_per_instance": 8,
    },
    experiment_name = "drone-images-reasoning-training-jobs",
    display_name = "drone-images-reasoning-train-batchsize-16,
    properties = {
        "_azureml.LogTrainingMetricsToAzMon": "true"
    },
    environment variables = {
        "KINETO_USE_DAEMON": "1",
        "ENABLE_AZUREML_TRAINING_PROFILER": "true",
        "AZUREML_PROFILER_WAIT_DURATION_SECOND": "2",
        "AZUREML_PROFILER_RUN_DURATION_MILLISECOND": "500",
        "AZUREML_COMMON_RUNTIME_USE_APPINSIGHTS_CAPABILITY": "true",
    }
}

# the following is for training
train_job = ml_client.jobs.create_or_update(job)
train_job

# the following registers the model which is required to deploy it to an endpoint.
model_output_path = f"azureml://jobs/{train_job.name}/outputs/mlflow_model_folder"
run_model = Model(
    path=model_output_path,
    name="grpo-reasoning-model",
    description=f"Model created from run {train_job.name}.",
    type=AssetTypes.MLFLOW_MODEL
)
ft_model = ml_client.models.create_or_update(run_model)
deployment = ManagedOnlineDeployment(
    name="grpo-rft-model-deployment",
    endpoint_name=online_endpoint_name,
    model=ft_model,
    instance_type="Standard_ND96amsr_A100_v4",
    instance_count=1
)
ml_client.begin_create_or_update(deployment)
   
