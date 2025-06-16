# %% imports
import os
import time
import types
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from ultralytics import YOLO
from ultralytics.nn import tasks

# ===================================================================
# PART 1: CUSTOM MODULE DEFINITIONS (Identical to your original)
# ===================================================================
class ScaleAwareFeatureAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mult_scale_heads = 8
        self.query_conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, stride=4)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
        self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)

    def forward(self, x, quarter_scale_x):
        orig_x = x
        x = self.query_conv1(x)
        x = self.query_conv2(x)
        quarter_scale_x = self.key_conv(quarter_scale_x)
        batch_size, C, roi_h, roi_w = x.size()
        x = x.view(batch_size, 1, C, roi_h, roi_w)
        quarter_scale_x = quarter_scale_x.view(batch_size, 1, C, roi_h, roi_w)
        x = torch.cat((x, quarter_scale_x), dim=1)
        batch_size, img_n, _, roi_h, roi_w = x.size()
        x_embed = x
        c_embed = x_embed.size(2)
        x_embed = x_embed.view(batch_size, img_n, self.mult_scale_heads, -1, roi_h, roi_w)
        target_x_embed = x_embed[:, [1]]
        ada_weights = torch.sum(x_embed * target_x_embed, dim=3, keepdim=True) / (float(c_embed / self.mult_scale_heads)**0.5)
        ada_weights = ada_weights.expand(-1, -1, -1, int(c_embed / self.mult_scale_heads), -1, -1).contiguous()
        ada_weights = ada_weights.view(batch_size, img_n, c_embed, roi_h, roi_w)
        ada_weights = ada_weights.softmax(dim=1)
        x = (x * ada_weights).sum(dim=1)
        aggregated_feature = F.interpolate(x, size=orig_x.shape[-2:], mode='bilinear', align_corners=False)
        aggregated_enhanced_representation = orig_x + aggregated_feature
        return aggregated_enhanced_representation

class FeatEnHancer(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatEnHancer, self).__init__()
        int_out_channels = 32
        out_channels = 24
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)
        self.ue_conv8 = nn.Conv2d(out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.ue_conv9 = nn.Conv2d(int_out_channels, 3, 3, 1, 1, bias=True)
        self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
        self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)
        self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=out_channels)
    
    def forward(self, x):
        original_x = x
        quarter_scale_x = self.quarter_conv(x); hexa_scale_x = self.hexa_conv(quarter_scale_x)
        x1 = self.relu(self.e_conv1(x)); quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x)); hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))
        x2 = self.relu(self.e_conv2(x1)); quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1)); hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))
        x3 = self.relu(self.e_conv3(x2)); quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2)); hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))
        x4 = self.relu(self.e_conv4(x3)); quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3)); hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1))); quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1))); hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1))); quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1))); hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))
        x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1))); quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1))); hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))
        x7 = self.scale_aware_aggregation(x7, quarter_scale_x7)
        hexa_scale_x7 = F.interpolate(hexa_scale_x7, size=x7.shape[-2:], mode='bilinear', align_corners=False)
        x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))
        enhancement_residual = torch.tanh(self.ue_conv9(x8))
        return original_x + enhancement_residual

# This new forward method will be used to capture the residual for visualization
def capture_residual_forward(self, x):
    original_x = x
    quarter_scale_x = self.quarter_conv(x); hexa_scale_x = self.hexa_conv(quarter_scale_x)
    x1 = self.relu(self.e_conv1(x)); quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x)); hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))
    x2 = self.relu(self.e_conv2(x1)); quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1)); hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))
    x3 = self.relu(self.e_conv3(x2)); quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2)); hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))
    x4 = self.relu(self.e_conv4(x3)); quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3)); hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))
    x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1))); quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1))); hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))
    x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1))); quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1))); hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))
    x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1))); quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1))); hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))
    x7 = self.scale_aware_aggregation(x7, quarter_scale_x7)
    hexa_scale_x7 = F.interpolate(hexa_scale_x7, size=x7.shape[-2:], mode='bilinear', align_corners=False)
    x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))
    enhancement_residual = torch.tanh(self.ue_conv9(x8))
    
    global enhancement_residual_tensor
    enhancement_residual_tensor = enhancement_residual
    return original_x + enhancement_residual

# ===================================================================
# PART 2: INJECT THE CUSTOM MODULE
# ===================================================================
setattr(tasks, 'FeatEnHancer', FeatEnHancer)

# Global variables to hold our tensors of interest
final_output_tensor = None
enhancement_residual_tensor = None

# Hook to capture the final output of the FeatEnHancer module
def final_output_hook(module, input, output):
    global final_output_tensor
    final_output_tensor = output

# %% main inference function
def main(args):
    # --- Setup ---
    run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(f"runs/detect/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_output_dir = output_dir / "final_enhanced_images"
    if args.save_final_output:
        final_output_dir.mkdir(exist_ok=True)
        
    residual_map_dir = output_dir / "enhancement_residual_maps"
    if args.save_residual_map:
        residual_map_dir.mkdir(exist_ok=True)

    print(f"Results will be saved to: {output_dir.resolve()}")
    
    # --- Setup Logging ---
    log_file_path = output_dir / "inference_log.txt"
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"--- YOLO Custom Inference Run ---\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model Path: {args.model_path}\nData Path: {args.data_path}\nConfidence: {args.conf}\nDevice: {args.device}\n\n")
        log_file.write("--- Per-Image Timings ---\n")

    # --- Model Loading ---
    print(f"Loading model from {args.model_path}...")
    model = YOLO(args.model_path)
    model.to(args.device)
    print("Model loaded successfully.")

    # --- Attach Hooks or Patch Method for Visualization ---
    feat_enhancer_module = None
    for module in model.model.modules():
        if isinstance(module, FeatEnHancer):
            feat_enhancer_module = module
            break

    if feat_enhancer_module:
        if args.save_final_output:
            feat_enhancer_module.register_forward_hook(final_output_hook)
            print("Attached hook to capture the final 'Enhanced Image'.")
        if args.save_residual_map:
            feat_enhancer_module.forward = types.MethodType(capture_residual_forward, feat_enhancer_module)
            print("Patched FeatEnHancer.forward method to capture the 'Enhancement Map'.")

    # --- Data Preparation ---
    image_dir = Path(args.data_path) / "images"
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    # --- Inference Loop ---
    inference_times = []
    global final_output_tensor, enhancement_residual_tensor
    print("\nStarting inference...")

    for i, img_path in enumerate(image_files):
        start_time = time.perf_counter()
        model(img_path, conf=args.conf, save=True, save_txt=args.save_txt, project=output_dir.parent.parent, name=output_dir.name)
        end_time = time.perf_counter()
        duration = end_time - start_time
        inference_times.append(duration)
        
        # --- Save Visualizations ---
        if args.save_final_output and final_output_tensor is not None:
            final_output_clamped = torch.clamp(final_output_tensor, 0, 1)
            save_path = final_output_dir / f"{img_path.stem}_enhanced_final.png"
            save_image(final_output_clamped, save_path)
        
        if args.save_residual_map and enhancement_residual_tensor is not None:
            residual_to_save = (enhancement_residual_tensor * 0.5) + 0.5
            save_path = residual_map_dir / f"{img_path.stem}_residual_map.png"
            save_image(residual_to_save, save_path)

        final_output_tensor, enhancement_residual_tensor = None, None
        
        log_message = f"Image {i+1}/{total_images}: {img_path.name} - Time: {duration:.4f} seconds"
        print(log_message)
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_message + "\n")

    print("\nInference complete.")

    # --- Performance Calculation & Final Logging (RESTORED) ---
    if inference_times:
        total_time = sum(inference_times)
        avg_time_per_image = total_time / total_images
        fps = 1 / avg_time_per_image if avg_time_per_image > 0 else 0

        summary_header = "\n--- Performance Summary ---\n"
        summary_total = f"Total images processed: {total_images}\n"
        summary_avg_time = f"Average inference time per image: {avg_time_per_image:.4f} seconds\n"
        summary_fps = f"Average FPS (Frames Per Second): {fps:.2f}\n"
        final_summary = summary_header + summary_total + summary_avg_time + summary_fps

        print(final_summary)
        with open(log_file_path, 'a') as log_file:
            log_file.write(final_summary)
    else:
        print("No images were processed, cannot calculate performance.")

# %% script execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO inference with custom modules, performance logging, and advanced visualization.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained custom YOLO model (.pt file).")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the test data directory (in YOLO format).")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold for detection.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run inference on (e.g., 'cuda:0', 'cpu').")
    
    # NEW, CORRECTED FLAGS (replaces --save-enhancement)
    parser.add_argument('--save-final-output', action='store_true', help="Save the final 'Enhanced Image' from the diagram (clamped).")
    parser.add_argument('--save-residual-map', action='store_true', help="Save the 'Enhancement Map' (the change itself) as a grayscale image.")
    
    # Restored flag
    parser.add_argument('--save-txt', action='store_true', help="Save predicted bounding boxes to .txt files.")

    args = parser.parse_args()
    main(args)