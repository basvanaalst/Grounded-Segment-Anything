# Download pre-trained weights for GroundingDINO after cd Grounded-Segment-Anything
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download weights for SAM
#! curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o sam_vit_h_4b8939.pth

# Install GroundingDINO, SAM and all other dependencies
#! python -m pip install -e segment_anything
#! python -m pip install -e GroundingDINO

# Optional for diffusion model and LLM stuff, might need one of these for the installation to work
#! pip install diffusers transformers accelerate scipy safetensors

import os
import numpy as np
import torch
import cv2
from PIL import Image
import torch

from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import load_model, annotate, load_image, predict
from segment_anything.segment_anything import SamPredictor, build_sam

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

input_dir = "images"
output_dir = "images_results"

# Building GroundingDINO inference model
groundingdino_model = load_model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
groundingdino_model = groundingdino_model.to(DEVICE)

print("dino done")

# Building SAM Model and SAM Predictor
try:
    sam = build_sam(checkpoint=SAM_CHECKPOINT_PATH)
    print("Model built successfully")
    sam.to(device=DEVICE)
    print("Model moved to device")
    sam_predictor = SamPredictor(sam)
    print("SamPredictor initialized")
except Exception as e:
    print(f"Error: {e}")

print("sam done")

#semantic segmentation for all files in images folder
for file in os.listdir(input_dir):
    if file.lower().endswith(".jpg"):
        TEXT_PROMPT = "plants"
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8

        input_path = os.path.join(input_dir, file)
        image_source, image = load_image(input_path)

        boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_THRESHOLD, 
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB

        # NMS post process
        #print(f"Before NMS: {len(detections.xyxy)} boxes")
        #nms_idx = torchvision.ops.nms(
        #    torch.from_numpy(detections.xyxy), 
        #    torch.from_numpy(detections.confidence), 
        #    NMS_THRESHOLD
        #).numpy().tolist()

        #detections.xyxy = detections.xyxy[nms_idx]
        #detections.confidence = detections.confidence[nms_idx]
        #detections.class_id = detections.class_id[nms_idx]

        #print(f"After NMS: {len(detections.xyxy)} boxes")

        sam_predictor.set_image(image_source)

        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )

        def show_masks(masks, image, random_color=True):
            annotated_frame_pil = Image.fromarray(image).convert("RGBA")

            for mask in masks:
                if random_color:
                    color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
                else:
                    color = np.array([30/255, 144/255, 255/255, 0.6])

                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
                annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)

            return np.array(annotated_frame_pil)

        annotated_frame_result = show_masks(masks, annotated_frame)

        # Assuming masks is a list or a tensor-like structure where masks[i][0] is a 2D mask
        mask_arrays = [masks[i][0].cpu().numpy() for i in range(len(masks))]

        # Stack masks into a single array (e.g., along the depth axis)
        combined_mask = np.stack(mask_arrays, axis=0)  # Shape: (num_masks, height, width)

        # Optionally, sum the masks or apply any other operation (e.g., max to avoid overlap intensity)
        final_mask = np.max(combined_mask, axis=0)  # This takes the maximum value at each pixel

        image_source_pil = Image.fromarray(image_source)
        annotated_frame_pil = Image.fromarray(annotated_frame_result)
        image_mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))

        original_array = np.array(image_source_pil.convert("RGBA"))
        original_array[:, :, 3] = image_mask_pil
        result_image = Image.fromarray(original_array)

        result_images = [(image_mask_pil, "mask"), (annotated_frame_pil, "annotated_frame"), (result_image, "result")]
        for _, (image, category) in enumerate(result_images):
            file_without_ext = os.path.splitext(file)[0]
            filename = f"{category}_{file_without_ext}.png"
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)
            print(f"Saved {output_path}")    


