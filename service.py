import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
import colorlog
import grpc
import cv2
import numpy as np
from concurrent import futures

#
from engins import InsightFaceRec, YoloDectect
from consts import PROJECT_NAME, PROJECT_DIR, LOG_PATH, SERVER_ADDRESS

#
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "protos"))
from protos import ai_engine_pb2_grpc, ai_engine_pb2

# 修改日志配置部分
logger = logging.getLogger(PROJECT_NAME)
logger.setLevel(logging.DEBUG)

# 先移除所有默认handler
logger.handlers.clear()

# 创建并添加彩色控制台handler
color_handler = logging.StreamHandler(sys.stdout)
color_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] \t%(message)s",
        datefmt="%x %X",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)
logger.addHandler(color_handler)

# 添加文件handler
file_handler = RotatingFileHandler(
    LOG_PATH, maxBytes=1000000, backupCount=1, encoding="utf-8"
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s", datefmt="%x %X")
)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


class AiEngineService(ai_engine_pb2_grpc.AiEngineServiceServicer):
    def __init__(self) -> None:
        self.yolo = YoloDectect()
        self.face_rec = InsightFaceRec()

    def YoloDetect(self, request, context):
        try:
            start = time.time()
            classes = request.classes
            stream = request.stream
            images = [
                cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                for image_data in request.images
            ]
            det_results = self.yolo.predict(
                source=images,
                classes=classes,
                stream=stream,
                batch=len(images),
            )
            simplified_results = []
            for result in det_results:
                item = result.summary()
                simplified_results.append(json.dumps(item))
            cost_time = time.time() - start
            logger.info(
                f"YoloDetect detect {len(simplified_results)} images, cost {cost_time:.2f}s"
            )
            return ai_engine_pb2.YoloDetResponse(detections=simplified_results)
        except Exception as e:
            errmsg = f"YoloDetect Error: {str(e)}"
            logger.error(errmsg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(errmsg)
            return ai_engine_pb2.YoloDetResponse()

    def InsightFaceRec(self, request, context):
        try:
            start = time.time()
            max_num = request.max_num
            image = cv2.imdecode(
                np.frombuffer(request.image, np.uint8), cv2.IMREAD_COLOR
            )
            faces = self.face_rec.faces(image, max_num=max_num)
            simplified_results = []
            for face in faces:
                item = {
                    "bbox": face.bbox.astype(float).tolist(),
                    "kps": face.kps.astype(float).tolist(),
                    "det_score": face.det_score.astype(float),
                    "embedding": face.embedding.astype(float).tolist(),
                }
                simplified_results.append(json.dumps(item))
            cost_time = time.time() - start
            logger.info(
                f"InsightFaceRec recognized {len(faces)} faces, cost {cost_time:.2f}s"
            )
            return ai_engine_pb2.InsightFaceRecResponse(faces=simplified_results)
        except Exception as e:
            errmsg = f"InsightFaceRec Error: {str(e)}"
            logger.error(errmsg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(errmsg)
            return ai_engine_pb2.InsightFaceRecResponse()


def serve():
    cpu_count = os.cpu_count() or 8
    server = grpc.server(
        thread_pool=futures.ThreadPoolExecutor(max_workers=cpu_count),
        options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )
    ai_engine_pb2_grpc.add_AiEngineServiceServicer_to_server(AiEngineService(), server)
    server.add_insecure_port(SERVER_ADDRESS)
    print(f"gRPC 服务启动在 {SERVER_ADDRESS}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
