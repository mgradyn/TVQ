import os
import numpy as np

from task_vectors import TaskVector, QuantizedTaskVector,QuantizedFinetunedModel, QuantizedBaseAndTaskVector
from eval import eval_single_dataset

def line_scaling(task_vector, alpha=0.0, beta=1.0, num_blocks=12):
    """
    Progressively scales the task vector based on layer depth.
    Parameters:
    -----------
    task_vector : dict
        A dictionary representing the residual between the fine-tuned checkpoint
        and the pre-trained checkpoint.
    alpha : float
         The minimum scaling factor for the blocks.
    beta : float
        The maximum scaling coefficient difference between the last and first block.
    num_blocks : int
        The total number of layer blocks in the model.
    Returns:
    --------
    scaled_task_vector : dict
        A copy of `task_vector` where each key is scaled based on the layer depth.
    """
    import copy
    # Deep copy the task vector to avoid modifying the original
    scaled_task_vector = copy.deepcopy(task_vector)
    # Generate the key blocks corresponding to the layers of the model
    key_blocks = [f".resblocks.{i}." for i in range(num_blocks)]
    # Create a scaling dictionary to store the scaling factor for each key
    scaling_dic = {}
    for k in task_vector.keys():
        # Find the layer block in the key and assign scaling factor based on layer depth
        for layer, block in enumerate(key_blocks):
            if block in k:
                scaling_dic[k] = alpha + beta * (layer / (num_blocks - 1))
                break
    print(f"LiNeS: The layers are scaled between {alpha} to {alpha + beta}")
    # Scale the task vector based on the scaling dictionary
    scaled_task_vector = {
        # Use alpha if layer is outside residual blocks
        k: task_vector[k] * scaling_dic.get(k, alpha)
        for k in task_vector.keys()
    }
    return scaled_task_vector


def run(args, log, exam_datasets, pretrained_checkpoint, source_root_path):
    model = args.model

    alpha = 1/8
    beta = 0.5

    task_vectors = []
    for dataset_name in exam_datasets:

        if args.load_tv_type == "baseline":
            tv = TaskVector(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints/{model}/{dataset_name}/finetuned.pt"
            )

        elif args.load_tv_type == "quantized_finetuned":
            assert args.load_task_bits is not None, "Specify load_task_bits for quantized_finetuned"
            tv = QuantizedFinetunedModel(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints_quantized{args.load_task_bits}bit/{model}/{dataset_name}/finetuned.pt"
            )
    
        elif args.load_tv_type == "quantized_task_vector":
            assert args.load_task_bits is not None, "Specify load_task_bits for quantized_task_vector"
            tv = QuantizedTaskVector(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints_taskvector_quantized{args.load_task_bits}bit/{model}/{dataset_name}/finetuned.pt"
            )
        
        elif args.load_tv_type == "quantized_residual_task_vector":
            assert args.load_base_bits is not None and args.load_task_bits is not None, \
                "Specify both load_base_bits and load_task_bits for quantized_residual_task_vector"
            tv = QuantizedBaseAndTaskVector(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints_taskvector_quantized{args.load_base_bits}bit/{model}/basevector.pt",
                f"{source_root_path}/checkpoints_base{args.load_base_bits}bit_restask{args.load_task_bits}bit/{model}/{dataset_name}/finetuned.pt"
            )

        scaled_task_vector = line_scaling(tv.vector, alpha=alpha, beta=beta, num_blocks=24)
        tv.vector = scaled_task_vector
        task_vectors.append(tv)

    task_vector_sum = sum(task_vectors)
    scaling_coef_ = 1.0
    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)
    log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

    accs = []
    for dataset in exam_datasets:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
        accs.append(metrics.get('top1')*100)
    log.info('Avg ACC:' + str(np.mean(accs)) + '%')
