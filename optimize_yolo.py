# """
# YOLO TensorRT Export and Basic Optimizations
# ------------------------------------------

# Key Steps:
# 1. Model Loading and Quantization (INT8/FP16)
#    - Reduces model size
#    - Speeds up inference
   
# 2. TensorRT Export
#    - Converts PyTorch model to TensorRT
#    - Optimizes for NVIDIA GPUs
   
# 3. Basic Pruning
#    - Removes redundant weights
#    - Makes model lighter
# """

import torch
import tensorrt as trt
from torch2trt import torch2trt
import torch.nn.utils.prune as prune

def optimize_and_export_yolo(model_path, save_path, input_size=(640, 640)):
    # Load model
    model = torch.load(model_path)
    model.eval()
    model.cuda()
    
    # Basic pruning (remove 20% of weights)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            
    # Create example input
    x = torch.randn(1, 3, *input_size).cuda()
    
    # Convert to TensorRT with FP16 precision
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        max_batch_size=1
    )
    
    # Save optimized model
    torch.save(model_trt.state_dict(), save_path)
    print(f"Optimized model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save optimized model')
    args = parser.parse_args()
    
    optimize_and_export_yolo(args.model, args.save_path)