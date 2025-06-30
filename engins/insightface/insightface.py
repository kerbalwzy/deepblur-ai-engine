import logging
import os
from typing import List
import cv2
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import DEFAULT_MP_NAME
from consts import PROJECT_NAME, ASSETS_DIR

logger = logging.getLogger(PROJECT_NAME)


class InsightFaceRec:

    def __init__(
        self,
        name=DEFAULT_MP_NAME,
        root=os.path.dirname(__file__),
        allowed_modules=["recognition", "detection"],
        ctx_id=0,  # -1 for CPU, 0 for GPU
        det_thresh: float = 0.45,  # threshold for detection
        det_size=(640, 640),  # image size for detection
        *args,
        **kwargs,
    ):
        self.app = FaceAnalysis(
            name=name, root=root, allowed_modules=allowed_modules, *args, **kwargs
        )
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
        logger.info("InsightFaceRec load model done")
        self.preheat()

    def preheat(self):
        """Preheat the model"""
        logger.info("InsightFaceRec preheat model......")
        preheat_img = cv2.imread(os.path.join(ASSETS_DIR, "person.jpg"))
        self.faces(img=preheat_img)

    def faces(self, img, max_num=0)->List[Face]:
        """Detect faces in an image"""
        faces = self.app.get(img, max_num=max_num)
        return faces
