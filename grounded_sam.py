#! python -m pip install -e segment_anything
#! python -m pip install -e GroundingDINO
#! pip install diffusers transformers accelerate scipy safetensors

import os, sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything.segment_anything import build_sam, SamPredictor 

from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

print("start_groundingDINO")
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

#! curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o sam_vit_h_4b8939.pth

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("start_SAM")
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

#semantic segmentation for all files in images folder
for file in os.listdir("images"):
    if file.endswith(".jpg"):
        print("image")
        TEXT_PROMPT = "plants"
        BOX_TRESHOLD = 0.30
        TEXT_TRESHOLD = 0.25

        image_source, image = load_image(file)

        boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB

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
            filename = f"{category}_{file}.png"
            output_path = os.path.join("images_results", filename)
            image.save(output_path)


