import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import time
import sys

sys.path.append('/home/TVQ/')

import torch
from args import parse_arguments
from quantization_utils import absmax_quantization, qunatization_error_check
import pickle

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD

model = 'ViT-B-32'
args = parse_arguments()
source_root_path = '/home/TVQ'
args.model = model
args.save = source_root_path+'/checkpoints/' + model
pretrained_checkpoint = source_root_path+'/checkpoints/'+model+'/zeroshot.pt'

avg_finetunedmodel_checkpoint = source_root_path+'/checkpoints/'+model+'/'

pretrained_state_dict = torch.load(pretrained_checkpoint, weights_only=False).state_dict()

# Initialize a dictionary to store the sum of all task vectors
finetuned_weight_sums = {}
for key, value in pretrained_state_dict.items():
    finetuned_weight_sums[key] = torch.zeros_like(value)

for dataset_name in exam_datasets:
    finetuned_checkpoint = source_root_path+'/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt'

    with torch.no_grad():
        print('TaskVector:' + finetuned_checkpoint)
        try:
            finetuned_state_dict = torch.load(finetuned_checkpoint, weights_only=False).state_dict()
        except:
            finetuned_state_dict = pickle.load(open(finetuned_checkpoint, 'rb')).state_dict()

        for key in pretrained_state_dict:
            if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                continue
            
            finetuned_weight = finetuned_state_dict[key]
            finetuned_weight_sums[key] += finetuned_weight  # Accumulate values

       
finetuned_weight_avg_state_dict = {key: finetuned_weight_sums[key] / len(exam_datasets) for key in finetuned_weight_sums}

os.makedirs(avg_finetunedmodel_checkpoint, exist_ok=True)
print ('saving...... '+ avg_finetunedmodel_checkpoint+f'/avg_ft_model.pt')
torch.save(finetuned_weight_avg_state_dict, avg_finetunedmodel_checkpoint+f'/avg_ft_model.pt')


            

