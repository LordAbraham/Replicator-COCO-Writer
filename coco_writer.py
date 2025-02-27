
import omni.replicator.core as rep
import warp as wp
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch
from omni.syntheticdata import SyntheticData
from numpy import uint8
from cv2 import findContours, RETR_CCOMP, CHAIN_APPROX_SIMPLE, contourArea
from io import BytesIO
from json import dumps as json_dumps
from datetime import datetime
from os.path import join as os_join

wp.init()
@wp.kernel
def mask_kernel(mask: wp.array(dtype=wp.uint8, ndim=3), # type: ignore
                color: wp.array(dtype=wp.uint8, ndim=1), # type: ignore
                instance_mask: wp.array(dtype=wp.uint8, ndim=2)):# type: ignore
    i, j = wp.tid()  # Get thread indices
    if (mask[i, j, 0] == color[0] and 
        mask[i, j, 1] == color[1] and 
        mask[i, j, 2] == color[2]):
        instance_mask[i, j] = wp.uint8(255)  # Explicit cast
    else:
        instance_mask[i, j] = wp.uint8(0)    # Explicit cast

class COCOWriter(Writer):
    def __init__(
                self,
                output_dir,
                rgb: bool = True,
                instance_segmentation: bool = False,
                image_output_format: str = "png",
                semantic_filter_predicate = None,
                bbox: bool = False,
                info: dict = None,
                licenses: dict = None,
                use_license: int = 0
            ):
        self.use_license = use_license
        if info is None:
            info = {
                "description": "Synthetic data",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y/%m/%d")
            }

        if licenses is None:
            licenses = [
                {
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "id": 0,
                    "name": "Attribution License"
                }
        ]
        self.coco = {
            "info": info,
            "licenses": licenses,
        }

        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._image_output_format = image_output_format
        self.instance_segmentation = instance_segmentation
        self.annotators = []

        if semantic_filter_predicate is not None:
            SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)
        # RGB
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        if bbox:
            self.annotators.append(AnnotatorRegistry.get_annotator("bounding_box_2d_tight"))
        if instance_segmentation:
            self.annotators.append(AnnotatorRegistry.get_annotator("instance_segmentation",
                                                                init_params={"colorize": True}))


    def create_coco_segmentation(self, labels, mask = None, first_annotation_id = 0):
        segmentations = {}
        for color in labels:
            height, width, _ = mask.shape
            d_mask = wp.array(mask, dtype=wp.uint8)
            d_color = wp.array(color[0], dtype=wp.uint8)
            d_instance_mask = wp.zeros((height, width), dtype=wp.uint8)
            wp.launch(kernel=mask_kernel, dim=(height, width), 
                    inputs=[d_mask, d_color, d_instance_mask])

            contours, hierarchy = findContours(d_instance_mask.numpy(), RETR_CCOMP, CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            
            segmentation = []
            area = 0.0
            for i, contour in enumerate(contours):
                area += contourArea(contour)
                contour = contour.squeeze().tolist()
                if len(contour) >= 6:  # COCO wymaga minimum 3 punktów (x,y)
                    segmentation.append([coord for point in contour for coord in point])
                    
            
            if segmentation:
                segmentations[color[2]] = {
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0
                }
                first_annotation_id += 1
                
        return segmentations

    def check_bbox_area(self, width, height, size_limit):
        area = height * width
        if area > size_limit:
            return True
        else:
            return False
            
    def get_bbox_data(self, data):
        bbox_data = {}
        bbox_ = data["data"]
        for i, _ in enumerate(bbox_):
            width = int(abs(bbox_['x_min'][i] - bbox_['x_max'][i]))
            height = int(abs(bbox_['y_min'][i] - bbox_['y_max'][i]))

            if not self.check_bbox_area(width, height , 1):
                continue

            if width != 2147483647 and height != 2147483647:
                bbox_data[data["info"]["primPaths"][i]]  = [int(bbox_['x_min'][i]), int(bbox_['y_min'][i]), width, height]
        return bbox_data

    def get_bbox_labels_categories(self, data):
        labels = []
        for i, key in enumerate(data["info"]["primPaths"].keys()):
            labels.append((None, data["data"]['semanticId'][i], key))
        categories = []
        for cat in data["info"]["idToLabels"].keys():
            categories.append({"supercategory": "object", "id": int(cat), "name": data["info"]["idToLabels"][cat]})
        return labels, categories
    
    def get_labels_and_categories(self, data):
        labels = []
        categories = []
        if "idToLabels" not in data:
            print(f"Warning: 'idToLabels' not found in data. Available keys: {list(data.keys())}")
            return labels, categories
        for color in data["idToLabels"].keys():
            if data["idToLabels"][color] == "BACKGROUND" or data["idToLabels"][color] == "UNLABELLED":
                continue
            if data["idToSemantics"][color]["class"] not in [category["name"] for category in categories]:
                categories.append({"supercategory": "object", "id": len(categories), "name": data["idToSemantics"][color]["class"]})
            category_id = next(category["id"] for category in categories if category["name"] == data["idToSemantics"][color]["class"])
            labels.append((tuple(map(int, color.strip('()').split(',')[:3])), category_id, data["idToLabels"][color]))
        return labels, categories

    def write(self, data):
        frame_id_str = f"{self._frame_id:04d}"
        names = []
        for key in data.keys():
            if key.startswith("rp_"):
                names.append(key.replace("rp_", "-"))
        if len(names) <= 1:
            names = [""]
        for name in names:
            bbox_data = {}
            labels = annotations = []
            segmentations = None
            coco = self.coco
            if "rgb" + name in data:

                filepath = os_join(f"{name[1:]}", "rgb", f"frame_{frame_id_str}.{self._image_output_format}")
                rgb_shape = data["rgb" + name].shape
                coco["images"] = [{"id": self._frame_id,
                                   "license": self.use_license, 
                                   "file_name": f"rgb/frame_{frame_id_str}.{self._image_output_format}", 
                                   "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                   "height": rgb_shape[0], 
                                   "width": rgb_shape[1]}]
                
                self._backend.write_image(filepath, data["rgb" + name])
            if "bounding_box_2d_tight" + name in data:
                bbox_data = self.get_bbox_data(data["bounding_box_2d_tight" + name])
                if not self.instance_segmentation:
                    labels, coco["categories"] = self.get_bbox_labels_categories(data["bounding_box_2d_tight" + name])

            if "instance_segmentation" + name in data:
                labels, coco["categories"] = self.get_labels_and_categories(data["instance_segmentation" + name]["info"])
                
                mask = data["instance_segmentation" + name]["data"]
                height, width = mask.shape[:2]
                mask = mask.view(uint8).reshape(height, width, -1)
                segmentations = self.create_coco_segmentation(labels, mask)

            for id, label in enumerate(labels):
                annotation = {}
                if segmentations is not None:
                    if label[2] not in segmentations:
                        continue
                    annotation = segmentations[label[2]]
                if label[2] in bbox_data:
                    annotation["bbox"] = bbox_data[label[2]]
                annotation["id"] = id
                annotation["category_id"] = label[1]
                annotation["image_id"] = self._frame_id
                annotations.append(annotation)

            coco["annotations"] = annotations
            coco_path = os_join(f"{name[1:]}", f"coco_{frame_id_str}.json")
            buf = BytesIO()
            buf.write(json_dumps(coco).encode())
            self._backend.write_blob(coco_path, buf.getvalue())

        self._frame_id += 1

rep.WriterRegistry.register(COCOWriter)