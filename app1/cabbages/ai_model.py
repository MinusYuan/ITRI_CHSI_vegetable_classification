import os
import cv2
import copy
import logging
import numpy as np
from PIL import Image
from typing import List, Optional
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import Colors
from sahi.utils.file import Path
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def visualize_object_predictions(
    image: np.array,
    object_prediction_list,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
    color: tuple = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    output_dir: Optional[str] = None,
    file_name: str = "prediction_visual",
    export_format: str = "png",
    new_class_label: Optional[dict] = None,
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.
    Arguments:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        hide_labels: hide labels
        hide_conf: hide confidence
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """    
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3

    # add masks to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        # visualize masks if present
        if object_prediction.mask is not None:
            # deepcopy mask so that original is not altered
            mask = object_prediction.mask.bool_mask
            # set color
            if colors is not None:
                color = colors(object_prediction.category.id)
            # draw mask
            rgb_mask = apply_color_mask(mask, color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)

    # add bboxes to image if present
    handle_lst = []
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()

        bbox = object_prediction.bbox.to_xyxy()
        category_name = object_prediction.category.name
        new_category_name = category_name.split('_')[0]
        category_id = new_class_label.get(new_category_name, object_prediction.category.id)
        score = object_prediction.score.value

        # set color
        if colors is not None:
            color = colors(category_id)
        # set bbox points
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # visualize boxes
        cv2.rectangle(
            image,
            p1,
            p2,
            color=color,
            thickness=rect_th,
        )

        if not hide_labels:
            # arange bounding box text location
            label = f"{category_name}"

            if not hide_conf:
                label += f" {score:.2f}"

            w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # add bounding box text
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )
        tmp_lst = [[v/255 for v in color], new_category_name]
        if tmp_lst not in handle_lst:
            handle_lst.append(tmp_lst)
    print(handle_lst, len(handle_lst))

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir) / (file_name + "." + export_format))
        # pil_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # cv2.imwrite(save_path, pil_image)
        plt.figure(figsize=(15, 18))
        plt.imshow(image)
        handles = [Rectangle((0, 0), 1, 1, color=c) for c, _ in handle_lst]
        labels = [n for _, n in handle_lst]
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(save_path)

    return {"image": image}

class SAHI(object):
    def __init__(self, model_path='cabbages/models/best.pt', model_type='yolov5', export_dir='media/output/', conf_thr=0.3):
        self.model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=conf_thr,
        )
        self.export_dir = export_dir
        self.new_mapping_tables = defaultdict(int)
        self.create_new_mapping_tables()
        
    def create_new_mapping_tables(self):
        for k, v in self.model.category_mapping.items():
            name = v.split('_')[0]
            self.new_mapping_tables[name] = self.new_mapping_tables.get(name, len(self.new_mapping_tables))

    def inference(self, image_path, slice=False, hide_labels=True, hide_conf=True):
        if slice: # detail-inference
            results = get_sliced_prediction(
                image_path,
                self.model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
        else: # fast-inference
            results = get_prediction(image_path, self.model)

        logging.info(f"image_path: {image_path}")
        origin_filename = os.path.basename(image_path)
        fn, _ = os.path.splitext(origin_filename)
        output_fn = f"{fn}_outputImage"
        logging.info(f"output_fn: {output_fn}")

        annotations = results.to_coco_annotations()
        logging.info(f"Len(annotations): {len(annotations)}")
        category_id_counts = {}
        for anno in annotations:
            category_name = anno['category_name'].split('_')[0]
            category_id_counts[category_name] = category_id_counts.get(category_name, 0) + 1

        # Save image (ext: png)
        # results.export_visuals(export_dir=self.export_dir, file_name=output_fn, hide_labels=hide_labels, hide_conf=hide_conf)
        visualize_object_predictions(
            image=np.ascontiguousarray(results.image),        	
            object_prediction_list=results.object_prediction_list,
            output_dir=self.export_dir,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
            file_name=output_fn,
            new_class_label=self.new_mapping_tables
        )

        # Remove original one
        if os.path.exists(image_path):
            os.remove(image_path)

        return category_id_counts, self.export_dir+output_fn+".png"

def get_SAHI_model(*kargs, **kwargs):
    return SAHI(*kargs, **kwargs)
