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

folder_path = "images\\"

im = cv2.imread("fox.jpg")


for file_path in file_path_list:

    cfg = get_cfg()

    # without cuda
    cfg.MODEL.DEVICE = "cpu"

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow("image", out.get_image()[:, :, ::-1])

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    from PIL import Image

    imbg = cv2.imread("background.png")

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
    cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :], mode='RGB')

    # Create a PIL image out of the mask
    mask = Image.fromarray((item_mask * 255).astype('uint8'))

    # Crop the mask to match the cropped image
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

    # Load in a background image and choose a paste position
    background = Image.fromarray(imbg, mode='RGB')
    paste_position = (x_min,y_min) #(300, 150)

    # Create a new foreground image as large as the composite and paste the cropped image on top
    new_fg_image = Image.new('RGB', background.size)
    new_fg_image.paste(cropped, paste_position)

    # Create a new alpha mask as large as the composite and paste the cropped mask
    new_alpha_mask = Image.new('L', background.size, color = 0)
    new_alpha_mask.paste(cropped_mask, paste_position)

    # Compose the foreground and background using the alpha mask
    composite = Image.composite(new_fg_image, background, new_alpha_mask)
    composite = np.asarray(composite)

    # Display the image
    #cv2.imshow(" ",np.array(composite))

    cv2.imwrite('crop_images.jpg', composite)

#cv2.imshow('crop_images.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()