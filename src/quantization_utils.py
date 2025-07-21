import torch


def absmax_quantization (X:torch.Tensor, qbit:int = 8):
    s = (2**(qbit-1)-1)/torch.max(torch.abs(X))

    X_q = (s*X).round()
    if qbit<=8:
        dtype = torch.int8
    elif qbit==16:
        dtype = torch.int16
  
    return X_q.to(dtype), s

def asymmetric_quantization(X: torch.Tensor, qbit: int = 8):
    """
    Perform asymmetric quantization on a given tensor using min-max scaling.

    Args:
        X (torch.Tensor): Input floating-point tensor.
        qbit (int): Bit width for quantization (default: 8-bit).

    Returns:
        X_q (torch.Tensor): Quantized integer tensor.
        scale (torch.Tensor): Scale factor.
        zero_point (torch.Tensor): Zero-point offset for dequantization.
    """
    X_min, X_max = X.min(), X.max()

    # Define quantization range
    qmin, qmax = 0, 2**qbit - 1  # Example: 8-bit → range [0, 255]

    # Compute scale and zero-point
    scale = (qmax - qmin) / (X_max - X_min) 
    zero_point =  -1* torch.round(scale * X_min)

    # Quantize: Round to nearest integer and clamp within range
    X_q = torch.round(scale * X + zero_point).clamp(qmin, qmax)
    
    if qbit<=8:
        dtype = torch.uint8
    elif qbit==16:
        dtype = torch.int16
    
    
    return X_q.to(dtype), scale, zero_point





def qunatization_error_check (original_state_dict, quantized_state_dict):
    accumulated_error = 0
    for key in original_state_dict.keys():
        weight_original = original_state_dict[key]
        weight_quantized = quantized_state_dict[key]
        if weight_quantized.dtype in [torch.int8]:
            if key + '_qscale' not in quantized_state_dict.keys():
                AssertionError('scale is missing for weight {}'.format(key))
            else:
                scale = quantized_state_dict[key + '_qscale']
            reconstructed_weight = weight_quantized.to(torch.float) / scale
        else:
            reconstructed_weight = weight_quantized
        error = weight_original - reconstructed_weight
        accumulated_error += torch.sum(torch.abs(error))#/torch.numel(error)
        # print(f'Error for weight {key}: {torch.max(torch.abs(error))}')
    print(f'accumuated Quantized error: {accumulated_error}')


def qunatization_error_check_asymmetric (original_state_dict, quantized_state_dict):
    accumulated_error = 0
    for key in original_state_dict.keys():
        weight_original = original_state_dict[key]
        weight_quantized = quantized_state_dict[key]
        if weight_quantized.dtype in [torch.uint8]:
            if key + '_qscale' not in quantized_state_dict.keys():
                AssertionError('scale is missing for weight {}'.format(key))
            else:
                scale = quantized_state_dict[key + '_qscale']
                zero_point = quantized_state_dict[key + '_qzeropoint']

            reconstructed_weight = (weight_quantized.to(torch.float)  -zero_point.to(torch.float)) / scale
        else:
            reconstructed_weight = weight_quantized
        error = weight_original - reconstructed_weight
        accumulated_error += torch.sum(torch.abs(error))#/torch.numel(error)
        # print(f'Error for weight {key}: {torch.max(torch.abs(error))}')
    print(f'accumuated Quantized error: {accumulated_error}')


