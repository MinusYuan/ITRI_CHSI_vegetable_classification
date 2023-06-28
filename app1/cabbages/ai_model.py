import os
import logging
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

class SAHI(object):
    def __init__(self, model_path='cabbages/models/best.pt', model_type='yolov5', export_dir='media/output/', conf_thr=0.3):
        self.model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=conf_thr,
        )
        self.export_dir = export_dir

    def inference(self, image_path, slice=False, hide_labels=True, hide_conf=True):
        def resize_output_image(filename, size=(800, 800)):
            original_file_path = os.path.join(self.export_dir, f"{filename}.png")
            try:
                im = Image.open(original_file_path)
                im.thumbnail(size)
                im.save(original_file_path)
            except IOError:
                print("cannot create thumbnail for", original_file_path)

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
        results.export_visuals(export_dir=self.export_dir, file_name=output_fn, hide_labels=hide_labels, hide_conf=hide_conf)

        # Resize it
        resize_output_image(output_fn)

        # Remove original one
        if os.path.exists(image_path):
            os.remove(image_path)

        return category_id_counts, self.export_dir+output_fn+".png"

def get_SAHI_model(*kargs, **kwargs):
    return SAHI(*kargs, **kwargs)
