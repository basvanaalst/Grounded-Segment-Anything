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
import supervision as sv
import torchvision

from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import Model
from segment_anything.segment_anything import SamPredictor, build_sam

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

input_dir = "images"
output_dir = "images_results2"

# Building GroundingDINO inference model
groundingdino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

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
        CLASSES = ["plants"]
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8

        input_path = os.path.join(input_dir, file)
        image = cv2.imread(input_path)

        # detect objects
        detections = groundingdino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, _ = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)


        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        image_source_pil = Image.fromarray(image)
        annotated_frame_pil = Image.fromarray(annotated_image)
        image_mask_pil = Image.fromarray((detections.mask[0] * 255).astype(np.uint8))
        
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


