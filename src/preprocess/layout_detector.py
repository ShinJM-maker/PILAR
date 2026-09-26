"""Layout detection using YOLOv8 trained on DocLayNet."""
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

# DocLayNet class mapping
DOCLAYNET_CLASSES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}

# Map DocLayNet classes to our BlockType
CLASS_TO_BLOCKTYPE = {
    "Title": "Title",
    "Section-header": "Header",
    "Text": "Paragraph",
    "Table": "Table",
    "Picture": "Figure",
    "Caption": "Caption",
    "List-item": "List",
    "Page-footer": "Footer",
    "Page-header": "Footer",
    "Footnote": "Footer",
    "Formula": "Paragraph",
}


class LayoutDetector:
    """DocLayNet-based layout detection."""

    def __init__(self, model_path: str = "yolov8x-doclaynet",
                 confidence_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.5,
                 device: str = "cuda:0"):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.model = None
        self._model_path = model_path

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from ultralytics import YOLO
            # Try loading from HuggingFace or local path
            self.model = YOLO(self._model_path)
        except Exception:
            # Fallback: download a DocLayNet YOLO model
            from ultralytics import YOLO
            self.model = YOLO("yolov8x.pt")  # base model as fallback
            print("WARNING: Using base YOLO model. Install DocLayNet weights for better results.")

    def detect(self, image: Image.Image) -> List[Dict]:
        """Detect layout elements in a page image.

        Returns:
            List of dicts with keys: bbox (x0,y0,x1,y1), class_name, block_type, confidence
        """
        self._load_model()
        img_array = np.array(image)

        results = self.model.predict(
            img_array,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    x0, y0, x1, y1 = box.xyxy[0].tolist()
                    class_name = DOCLAYNET_CLASSES.get(cls_id, "Text")
                    block_type = CLASS_TO_BLOCKTYPE.get(class_name, "Paragraph")
                    detections.append({
                        "bbox": (x0, y0, x1, y1),
                        "class_name": class_name,
                        "block_type": block_type,
                        "confidence": conf,
                    })

        # Sort by reading order: top-to-bottom, then left-to-right
        detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return detections

    def detect_from_path(self, image_path: str | Path) -> List[Dict]:
        image = Image.open(image_path).convert("RGB")
        return self.detect(image)
