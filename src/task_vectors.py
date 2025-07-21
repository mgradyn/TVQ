import torch
import pickle


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                print('TaskVector:' + finetuned_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                try:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                except:
                    finetuned_state_dict = pickle.load(open(finetuned_checkpoint, 'rb')).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


class QuantizedTaskVector(TaskVector):
    def __init__(self, pretrained_checkpoint=None, quantized_task_vector_checkpoint=None, vector=None):
        print ('Using Quantized TaskVector >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print (quantized_task_vector_checkpoint)

        if vector is not None:
            self.vector = vector
        else:
            assert quantized_task_vector_checkpoint is not None 
            with torch.no_grad():
                print('TaskVector:' + quantized_task_vector_checkpoint)
                quantized_state_dict = torch.load(quantized_task_vector_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

                self.vector = {}

                for key in pretrained_state_dict.keys():
                    weight_quantized = quantized_state_dict[key]
                    if weight_quantized.dtype in [torch.int8, torch.uint8]:
                        if key + '_qscale' not in quantized_state_dict.keys():
                            AssertionError('scale is missing for weight {}'.format(key))
                        else:
                            scale = quantized_state_dict[key + '_qscale']
                        
                        if key + '_qzeropoint' in quantized_state_dict.keys():
                            zero_point = quantized_state_dict[key + '_qzeropoint']
                            weight_quantized = weight_quantized- zero_point

                        reconstructed_taskvector = weight_quantized / scale
                    else:
                        reconstructed_taskvector = weight_quantized
                
                    self.vector[key] = reconstructed_taskvector


class QuantizedFinetunedModel(TaskVector):
    def __init__(self, pretrained_checkpoint=None, quantized_finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        print ('Using Quantized Finetuned model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print (quantized_finetuned_checkpoint)

        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and quantized_finetuned_checkpoint is not None
            with torch.no_grad():
                print('TaskVector:' + quantized_finetuned_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                quantized_finetuned_state_dict = torch.load(quantized_finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict.keys():
                    weight_quantized = quantized_finetuned_state_dict[key]
                    if weight_quantized.dtype in [torch.int8, torch.uint8]:
                        if key + '_qscale' not in quantized_finetuned_state_dict.keys():
                            AssertionError('scale is missing for weight {}'.format(key))
                        else:
                            scale = quantized_finetuned_state_dict[key + '_qscale']
                            zero_point = quantized_finetuned_state_dict[key + '_qzeropoint']
                        reconstructed_finetuned_weight = (weight_quantized.to(torch.float)  -zero_point.to(torch.float)) / scale
                    else:
                        reconstructed_finetuned_weight = weight_quantized
                
                    self.vector[key] = reconstructed_finetuned_weight - pretrained_state_dict[key]


class QuantizedBaseAndTaskVector(TaskVector):
    def __init__(self, pretrained_checkpoint=None, quantized_base_vector_checkpoint = None, quantized_task_vector_checkpoint=None, vector=None):
        print ('Using Quantized Base + TaskVector >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print (f"quantized_base_vector_checkpoint {quantized_base_vector_checkpoint}")
        print (f"quantized_task_vector_checkpoint {quantized_task_vector_checkpoint}")

        if vector is not None:
            self.vector = vector
        else:
            assert quantized_task_vector_checkpoint is not None 
            with torch.no_grad():
                print('TaskVector:' + quantized_task_vector_checkpoint)
                quantized_task_vector_state_dict = torch.load(quantized_task_vector_checkpoint)
                quantized_base_vector_state_dict = torch.load(quantized_base_vector_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

                self.vector = {}

                for key in pretrained_state_dict.keys():

                    base_quantized = quantized_base_vector_state_dict[key]
                    if base_quantized.dtype in [torch.int8, torch.uint8]:
                        if key + '_qscale' not in quantized_base_vector_state_dict.keys():
                            AssertionError('scale is missing for weight {}'.format(key))
                        else:
                            scale = quantized_base_vector_state_dict[key + '_qscale']
                        
                        if key + '_qzeropoint' in quantized_base_vector_state_dict.keys():
                            zero_point = quantized_base_vector_state_dict[key + '_qzeropoint']
                            base_quantized = base_quantized- zero_point
                            
                        reconstructed_basevector = base_quantized / scale
                    else:
                        reconstructed_basevector = base_quantized


                    weight_quantized = quantized_task_vector_state_dict[key]
                    if weight_quantized.dtype in [torch.int8, torch.uint8]:
                        if key + '_qscale' not in quantized_task_vector_state_dict.keys():
                            AssertionError('scale is missing for weight {}'.format(key))
                        else:
                            scale = quantized_task_vector_state_dict[key + '_qscale']
                        
                        if key + '_qzeropoint' in quantized_task_vector_state_dict.keys():
                            zero_point = quantized_task_vector_state_dict[key + '_qzeropoint']
                            weight_quantized = weight_quantized- zero_point
                            
                        reconstructed_taskvector = weight_quantized / scale
                    else:
                        reconstructed_taskvector = weight_quantized
                                       
                
                    self.vector[key] = reconstructed_basevector + reconstructed_taskvector