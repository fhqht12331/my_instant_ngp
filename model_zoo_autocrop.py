# check pytorch installation:
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from PIL import Image

src_path = "C:\test\t4\Instant-NGP\images"
dst_path = "C:\test\t4\tt\images"

imbg = cv2.imread("none.png")


def crop_images(src_path, dst_path):
    # Set up configuration and predictor
    cfg = get_cfg()

    # without cuda
    cfg.MODEL.DEVICE = "cpu"

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    # Load images from source path
    file_names = os.listdir(src_path)
    file_names = [f for f in file_names if f.endswith(".jpg")]
    file_names.sort()

    for i, file_name in enumerate(file_names):
        file_path = os.path.join(src_path, file_name)
        image = cv2.imread(file_path)
        height, width, _ = image.shape

        # Run object detection to get bounding box coordinates
        outputs = predictor(image)
        instances = outputs["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        class_ids = instances.pred_classes.cpu().numpy()

        # Crop images using bounding boxes
        for j, box in enumerate(boxes):
        # Get the masks
            masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))

            # Pick an item to mask
            item_mask = masks[0]

            # Get the true bounding box of the mask (not the same as the bbox prediction)
            segmentation = np.where(item_mask == True)
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            print(x_min, x_max, y_min, y_max)

            # Create a cropped image from just the portion of the image we want
            cropped = Image.fromarray(image[y_min:y_max, x_min:x_max, :], mode='RGB')
            # Create a PIL image out of the mask
            mask = Image.fromarray((item_mask * 255).astype('uint8'))

            # Crop and save the image
            cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

            # Load in a background image and choose a paste position
            background = Image.fromarray(imbg, mode='RGB')
            paste_position = (x_min,y_min) #(300, 150)

            # Create a new foreground image as large as the composite and paste the cropped image on top
            new_fg_image = Image.new('RGB', background.size)
            new_fg_image.paste(cropped)

            # Create a new alpha mask as large as the composite and paste the cropped mask
            new_alpha_mask = Image.new('L', background.size, color = 0)
            new_alpha_mask.paste(cropped_mask)

            # Compose the foreground and background using the alpha mask
            composite = Image.composite(new_fg_image, cropped, new_alpha_mask)
            cropped_image = np.asarray(composite)

            crop_name = f"{os.path.splitext(file_name)[0]}_{j:03}.jpg"
            crop_path = os.path.join(dst_path, crop_name)
            cv2.imwrite(crop_path, cropped_image)


if __name__ == "__main__":
    src_path = "images"
    dst_path = "crop_images"
    os.makedirs(dst_path, exist_ok=True)
    crop_images(src_path, dst_path)
    print("Done!")