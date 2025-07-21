import os
import numpy as np

from task_vectors import TaskVector, QuantizedTaskVector,QuantizedFinetunedModel, QuantizedBaseAndTaskVector
from eval import eval_single_dataset


def run(args, log, exam_datasets, pretrained_checkpoint, source_root_path):
    model = args.model

    # Determine how to construct the task vectors
    if args.load_tv_type == "baseline":
        task_vectors = [
            TaskVector(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints/{model}/{dataset_name}/finetuned.pt"
            )
            for dataset_name in exam_datasets
        ]

    elif args.load_tv_type == "quantized_finetuned":
        assert args.load_task_bits is not None, "Specify load_task_bits for quantized_finetuned"
        task_vectors = [
            QuantizedFinetunedModel(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints_quantized{args.load_task_bits}bit/{model}/{dataset_name}/finetuned.pt"
            )
            for dataset_name in exam_datasets
        ]

    elif args.load_tv_type == "quantized_task_vector":
        assert args.load_task_bits is not None, "Specify load_task_bits for quantized_task_vector"
        task_vectors = [
            QuantizedTaskVector(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints_taskvector_quantized{args.load_task_bits}bit/{model}/{dataset_name}/finetuned.pt"
            )
            for dataset_name in exam_datasets
        ]

    elif args.load_tv_type == "quantized_residual_task_vector":
        assert args.load_base_bits is not None and args.load_task_bits is not None, \
            "Specify both load_base_bits and load_task_bits for quantized_residual_task_vector"
        task_vectors = [
            QuantizedBaseAndTaskVector(
                pretrained_checkpoint,
                f"{source_root_path}/checkpoints_taskvector_quantized{args.load_base_bits}bit/{model}/basevector.pt",
                f"{source_root_path}/checkpoints_base{args.load_base_bits}bit_restask{args.load_task_bits}bit/{model}/{dataset_name}/finetuned.pt"
            )
            for dataset_name in exam_datasets
        ]

    else:
        raise ValueError(f"Unsupported load_tv_type: {args.load_tv_type}")



    task_vector_sum = sum(task_vectors)

    scaling_coef_ = 0.3

    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)
    log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

    accs = []
    for dataset in exam_datasets:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
        accs.append(metrics.get('top1')*100)

    log.info('Avg ACC:' + str(np.mean(accs)) + '%')
