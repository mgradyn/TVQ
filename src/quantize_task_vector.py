import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time
import sys
import tqdm
from enum import Enum

sys.path.append('/home/TVQ/')

import torch
from task_vectors import TaskVector
from args import parse_arguments
from quantization_utils import asymmetric_quantization, qunatization_error_check_asymmetric
import matplotlib.pyplot as plt
import pickle

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] 

model = 'ViT-B-32'
args = parse_arguments()
vis_histogram = False

source_root_path = '/home/TVQ'
args.model = model
args.save = source_root_path+'/checkpoints/' + model
args.logs_path = source_root_path+'/logs/' + model
pretrained_checkpoint = source_root_path+'/checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

class QuantizationTargetModel(Enum):
    TASK_VECTOR = "task_vector"
    FINETUNED_MODEL = "finetuned"


###### Quantization configuration. ##################################################################################
"""
TARGET_QUANTIZED_MODEL: finetuned model or task vector?  
RESIDUAL_TASKVECTOR: advanced quantization (our main quantization method). 
QUANTIZATION_BIT: how many bits you want to quantize to? (for residual vector / taskvector / finetuned model)
QUANTIZATION_BIT_FOR_BASEVECTOR: how many bits you want to quantize to? (for base vector)
BASE_Q_ERROR_CORRECTION: correcting quantization error for base vector
"""
TARGET_QUANTIZED_MODEL = QuantizationTargetModel(args.quantize_target)
RESIDUAL_TASKVECTOR = args.quantize_residual
QUANTIZATION_BIT = args.quantize_task_bit
QUANTIZATION_BIT_FOR_BASEVECTOR = args.quantize_base_bit
BASE_Q_ERROR_CORRECTION = args.q_error_correction
#####################################################################################################################

pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
reference_state_dict = pretrained_state_dict



if RESIDUAL_TASKVECTOR:
    quantized_base_checkpoint = source_root_path+'/checkpoints_taskvector_quantized'+str(QUANTIZATION_BIT_FOR_BASEVECTOR)+'bit'+'/'+model
    avg_ft_model_checkpoint = source_root_path+'/checkpoints/'+model+f'/avg_ft_model.pt'
    
    avg_ft_model_state_dict = torch.load(avg_ft_model_checkpoint)
    reference_state_dict = avg_ft_model_state_dict
    base_vector_state_dict = {}
    quantized_base_vector_state_dict = {}
    dequantized_base_vector_state_dict = {} # use [dequantized_base_vector_state_dict] + [pretrained_state_dict] to get [avg_ft_model_state_dict] (with quantization error)
    with torch.no_grad():
        print('Base vector: Pretrain ----> Finetuned avg model')

        for key in pretrained_state_dict:
            base_vector = avg_ft_model_state_dict[key] - pretrained_state_dict[key]
            base_vector_state_dict[key] = base_vector

            if ('weight' not in key) or ('ln_' in key) or ('token_embedding.weight' in key):
                quantized_base_vector_state_dict[key] = base_vector
                dequantized_base_vector_state_dict[key] = base_vector
                continue

            W_q, scale, zero_point = asymmetric_quantization(base_vector, qbit=QUANTIZATION_BIT_FOR_BASEVECTOR) 
            quantized_base_vector_state_dict[key] = W_q
            quantized_base_vector_state_dict[key + '_qscale'] = scale
            quantized_base_vector_state_dict[key + '_qzeropoint'] = zero_point

            dequantized_base_vector_state_dict[key] = (W_q.to(torch.float)  -zero_point.to(torch.float)) / scale # the reason why we use dequantized base vector is to contain qunatization err when we compute residual task vector

    qunatization_error_check_asymmetric(base_vector_state_dict, quantized_base_vector_state_dict)
    os.makedirs(quantized_base_checkpoint, exist_ok=True)
    torch.save(quantized_base_vector_state_dict, quantized_base_checkpoint+f'/basevector.pt')

    # (quantized-corrected base vector): make reference state dict as [dequantized_base_vector_state_dict] + [pretrained_state_dict] 
    if BASE_Q_ERROR_CORRECTION:
        reference_state_dict = {}
        for key in pretrained_state_dict:
            reference_state_dict[key] = dequantized_base_vector_state_dict[key] + pretrained_state_dict[key]
    



for dataset_name in exam_datasets:
    finetuned_checkpoint = source_root_path+'/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt'
    if TARGET_QUANTIZED_MODEL == QuantizationTargetModel.TASK_VECTOR:
        if RESIDUAL_TASKVECTOR:
            quantized_target_checkpoint = source_root_path+'/checkpoints_base'+str(QUANTIZATION_BIT_FOR_BASEVECTOR)+'bit_'+f'restask{QUANTIZATION_BIT}bit'+'/'+model+'/'+dataset_name
        else:
            quantized_target_checkpoint = source_root_path+'/checkpoints_taskvector_quantized'+str(QUANTIZATION_BIT)+'bit'+'/'+model+'/'+dataset_name
    elif TARGET_QUANTIZED_MODEL == QuantizationTargetModel.FINETUNED_MODEL:
        quantized_target_checkpoint = source_root_path+'/checkpoints_quantized'+str(QUANTIZATION_BIT)+'bit'+'/'+model+'/'+dataset_name
    else:
        AssertionError('Invalid quantization target model')
        
    quantized_task_vector_state_dict = {}
    quantized_finetuned_state_dict = {}
    task_vector_state_dict = {}

    with torch.no_grad():
        print('TaskVector:' + finetuned_checkpoint)
        try:
            finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
        except:
            finetuned_state_dict = pickle.load(open(finetuned_checkpoint, 'rb')).state_dict()

        for key in reference_state_dict:
            task_vector_weight = finetuned_state_dict[key] - reference_state_dict[key]
            task_vector_state_dict[key] = task_vector_weight

            if ('weight' not in key) or ('ln_' in key) or ('token_embedding.weight' in key):
                quantized_task_vector_state_dict[key] = task_vector_weight
                quantized_finetuned_state_dict[key] = finetuned_state_dict[key]
                continue
            
            if vis_histogram:
                print (task_vector_weight.min(), task_vector_weight.max(), task_vector_weight.size())
                print (finetuned_state_dict[key].min(), finetuned_state_dict[key].max(), finetuned_state_dict[key].size())

                # Create a figure and two subplots in a row
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Plot first histogram
                axes[0].hist(task_vector_weight.flatten(), bins=30, color='blue', alpha=0.7, edgecolor='black')
                axes[0].set_title("Histogram of task_vector [Conv1.weight]")
                axes[0].set_xlabel("Weight")
                axes[0].set_ylabel("Frequency")

                # Plot second histogram
                axes[1].hist(finetuned_state_dict[key].flatten(), bins=30, color='green', alpha=0.7, edgecolor='black')
                axes[1].set_title("Histogram of Finetuned Model Weight [Conv1.weight]")
                axes[1].set_xlabel("Weight")
                axes[1].set_ylabel("Frequency")

                # Adjust layout for better spacing
                plt.tight_layout()
                plt.savefig(f'./taskvec_vs_finetuned_{key}.png')
            
            if  TARGET_QUANTIZED_MODEL == QuantizationTargetModel.TASK_VECTOR:
                W_q, scale, zero_point = asymmetric_quantization(task_vector_weight, qbit=QUANTIZATION_BIT) 
                quantized_task_vector_state_dict[key] = W_q
                quantized_task_vector_state_dict[key + '_qscale'] = scale
                quantized_task_vector_state_dict[key + '_qzeropoint'] = zero_point
            else:
                W_q, scale, zero_point  = asymmetric_quantization(finetuned_state_dict[key], qbit=QUANTIZATION_BIT) 
                quantized_finetuned_state_dict[key] = W_q
                quantized_finetuned_state_dict[key + '_qscale'] = scale
                quantized_finetuned_state_dict[key + '_qzeropoint'] = zero_point

        # qunatization_error_check_asymmetric(task_vector_state_dict, quantized_task_vector_state_dict)

        if TARGET_QUANTIZED_MODEL == QuantizationTargetModel.TASK_VECTOR:
            qunatization_error_check_asymmetric(task_vector_state_dict, quantized_task_vector_state_dict)
            os.makedirs(quantized_target_checkpoint, exist_ok=True)
            torch.save(quantized_task_vector_state_dict, quantized_target_checkpoint+'/finetuned.pt')
        else:
            qunatization_error_check_asymmetric(finetuned_state_dict, quantized_finetuned_state_dict)
            os.makedirs(quantized_target_checkpoint, exist_ok=True)
            torch.save(quantized_finetuned_state_dict, quantized_target_checkpoint+'/finetuned.pt')