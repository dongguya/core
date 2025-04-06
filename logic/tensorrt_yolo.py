import os
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import atexit  # context 정리용

# 설정값
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dongguya.engine")
IMG_SIZE = 640
NUM_KEYPOINTS = 24
NUM_CLASSES = 3

# TensorRT 로깅
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 0. CUDA 초기화 및 context 생성
cuda.init()
device = cuda.Device(0)
cuda_context = device.make_context()
atexit.register(cuda_context.pop)  # 안전한 종료 시 context 반납

# 1. 엔진 로딩
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(MODEL_PATH)
if engine is None:
    raise RuntimeError("TensorRT 엔진 로딩 실패")

context = engine.create_execution_context()

# 2. CUDA I/O 메모리 설정
input_shape = (1, 3, IMG_SIZE, IMG_SIZE)
output_shape = (1, (4 + NUM_CLASSES + NUM_KEYPOINTS * 3), 8400)  # (1, 79, 8400)
input_nbytes = int(np.prod(input_shape) * np.float32().nbytes)
output_nbytes = int(np.prod(output_shape) * np.float32().nbytes)

d_input = cuda.mem_alloc(input_nbytes)
d_output = cuda.mem_alloc(output_nbytes)
bindings = [int(d_input), int(d_output)]

# 3. 전처리 함수
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32, copy=False)

# 4. 후처리 함수
def postprocess(output, orig_shape, conf_thres=0.5, iou_thres=0.5):
    preds = output.reshape((4 + NUM_CLASSES + NUM_KEYPOINTS * 3, 8400)).T
    cls_conf = preds[:, 4:4 + NUM_CLASSES]
    cls_scores = cls_conf.max(axis=1)
    cls_ids = cls_conf.argmax(axis=1)
    mask = cls_scores > conf_thres
    if not np.any(mask):
        return []

    boxes = preds[mask, :4]
    scores = cls_scores[mask]
    classes = cls_ids[mask]
    keypoints = preds[mask, 4 + NUM_CLASSES:].reshape(-1, NUM_KEYPOINTS, 3)

    xy = boxes[:, :2] - boxes[:, 2:] / 2
    wh = boxes[:, :2] + boxes[:, 2:]
    boxes_xyxy = np.concatenate((xy, wh), axis=1)

    # 간단한 NMS
    keep = []
    idxs = scores.argsort()[::-1]
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        ious = compute_iou_np(boxes_xyxy[current], boxes_xyxy[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]

    results = []
    h, w = orig_shape
    scale = np.array([w / IMG_SIZE, h / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE])
    for i in keep:
        box = boxes_xyxy[i] * scale
        kps = keypoints[i]
        kps[:, 0] *= w / IMG_SIZE
        kps[:, 1] *= h / IMG_SIZE
        results.append({
            "box": box.astype(int).tolist(),
            "score": float(scores[i]),
            "class": int(classes[i]),
            "keypoints": kps
        })
    return results

# IoU 계산 함수
def compute_iou_np(box1, boxes2):
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

# 5. 추론 함수
def infer(frame):
    orig_shape = frame.shape[:2]
    img_input = preprocess_frame(frame)
    img_input = np.ascontiguousarray(img_input, dtype=np.float32)

    context.set_binding_shape(0, input_shape)  # 필수!

    cuda.memcpy_htod(d_input, img_input)

    output = np.zeros(output_shape, dtype=np.float32)  # 선언 위치 이동
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(output, d_output)

    # 디버깅용 (선택)
    print(f"Output max: {output.max()}, min: {output.min()}, mean: {output.mean()}")

    return postprocess(output, orig_shape)
