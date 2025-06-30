__all__ = ["Yolo11nSegTRT"]
import logging
import os
import shutil
from ultralytics import YOLO

#
from consts import PROJECT_NAME, ASSETS_DIR


logger = logging.getLogger(PROJECT_NAME)


class YoloDectect:

    def __init__(self, model_size: str = "m") -> None:
        assert model_size in ["n", "s", "m", "l", "x"], Exception(
            f"Param model_size must be in ['n', 's', 'm', 'l', 'x']"
        )
        model_path = os.path.join(os.path.dirname(__file__), f"yolo11{model_size}.onnx")
        if not os.path.exists(model_path):
            logger.warn(f"transfer model yolo11{model_size}.onnx......")
            pt_model = YOLO(f"yolo11{model_size}.pt", verbose=True)
            exported_path = pt_model.export(
                format="onnx",  # export as engine file
                device=0,  # use the first GPU
                half=True,  # use FP16 precision
                workspace=4,  # set workspace size
            )
            shutil.move(exported_path, model_path)
            os.remove(os.path.join(os.getcwd(), f"yolo11{model_size}.pt"))
            del pt_model
            logger.warn(f"transfer model yolo11{model_size}.onnx done")
        self.model = YOLO(model_path, task="detect")
        logger.info("YoloDectect load model done")
        self.preheat()

    def preheat(self):
        logger.info("YoloDectect preheat model......")
        preheat_img = os.path.join(ASSETS_DIR, "person.jpg")
        self.predict(source=preheat_img)

    def predict(
        self,
        source=None,
        classes=[0, 2, 7],
        conf=0.45,
        iou=0.45,
        device=0,
        stream=False,
        half=True,
        verbose=False,
        *arg,
        **kwargs,
    ):
        """
        args:
            source: 0 for webcam, 'path/to/img.jpg', 'path/to/video.mp4', 'rtsp://example.com/media.mp4'
            classes: list of class ids, default is [0, 2, 7] for person and car and truck
            batch: batch size, default is 1
            conf: object confidence threshold, default is 0.35
            iou: NMS IoU threshold, default is 0.35
            device: cuda device, i.e. 0 or 0,1,2,3 or cpu, default is 0
            imgsz: inference size (pixels), uses size=640 if size<1280, uses size=1280 if size>1280, default is 640
            stream: True/False for webcam streaming, default is False
            half: True/False for half precision, default is True
            verbose: True/False for verbose output, default is False
        """
        return self.model.predict(
            source=source,
            classes=classes,
            conf=conf,
            iou=iou,
            device=device,
            stream=stream,
            half=half,
            verbose=verbose,
            *arg,
            **kwargs,
        )
