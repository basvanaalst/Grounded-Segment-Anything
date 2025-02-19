# Download pre-trained weights for GroundingDINO after cd Grounded-Segment-Anything
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download weights for SAM
#! curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o sam_vit_h_4b8939.pth

# Install GroundingDINO, SAM and all other dependencies
#! python -m pip install -e segment_anything
#! python -m pip install -e GroundingDINO

# Optional for diffusion model and LLM stuff, might need one of these for the installation to work
#! pip install diffusers transformers accelerate scipy safetensors

#! pip install segment-anything-hq

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
from segment_anything_hq import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

input_dir = "images"
output_dir = "images_results_hq"

# Building GroundingDINO
groundingdino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

print("dino done")

# Building SAM
light_hqsam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
light_hqsam.to(DEVICE)
sam_predictor = SamPredictor(light_hqsam)

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

        detections = groundingdino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

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

        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # change color of masks!

        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
       
        mask_arrays = [detections.mask[i] for i in range(len(detections.mask))]
        combined_mask = np.stack(mask_arrays, axis=0)
        final_mask = np.max(combined_mask, axis=0) 
        mask_image = (final_mask * 255).astype(np.uint8)

        image_source = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) if image.shape[2] == 3 else image.copy()
        annotated_frame = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2BGRA) if annotated_image.shape[2] == 3 else annotated_image.copy()

        result_image = image_source.copy()
        result_image[:, :, 3] = mask_image

        result_images = [(mask_image, "mask"), (annotated_frame, "annotated_frame"), (result_image, "result")]
        for image, category in result_images:
            file_without_ext = os.path.splitext(file)[0]
            filename = f"{category}_{file_without_ext}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            print(f"Saved {output_path}")

