import os
import numpy as np
import torch

from task_vectors import TaskVector, QuantizedTaskVector,QuantizedFinetunedModel, QuantizedBaseAndTaskVector
from eval import eval_single_dataset

def apply_vector(vector, pretrained_checkpoint):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model


def emr_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers

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


    # merge models
    vector_unified, masks, rescalers = emr_merge(task_vectors)

    accs = []
    for i, dataset in enumerate(exam_datasets):
        task_vector_recon = {}
        for n in vector_unified:
            task_vector_recon[n] =  vector_unified[n] * masks[n][i] * rescalers[i]
        image_encoder = apply_vector(task_vector_recon, pretrained_checkpoint)
        metrics = eval_single_dataset(image_encoder, dataset, args)
        log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
        accs.append(metrics.get('top1')*100)
    log.info('Avg ACC:' + str(np.mean(accs)) + '%')
