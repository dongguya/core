import os
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

# 설정값
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dongguya.engine")
IMG_SIZE = 640
NUM_KEYPOINTS = 24
NUM_CLASSES = 3

# TensorRT 로깅
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. 엔진 로딩
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(MODEL_PATH)
context = engine.create_execution_context()

# 2. CUDA I/O 메모리 설정
input_shape = (1, 3, IMG_SIZE, IMG_SIZE)
output_shape = (1, (4 + NUM_CLASSES + NUM_KEYPOINTS * 3), 8400)  # (1, 79, 8400)
input_nbytes = np.prod(input_shape) * np.float32().nbytes
output_nbytes = np.prod(output_shape) * np.float32().nbytes

d_input = cuda.mem_alloc(input_nbytes)
d_output = cuda.mem_alloc(output_nbytes)

bindings = [int(d_input), int(d_output)]

# 3. 프레임 전처리
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

# 4. 후처리 로직 (PyTorch 없이 NumPy로)
def postprocess(output, orig_shape, conf_thres=0.5, iou_thres=0.5):
    preds = output.reshape((4 + NUM_CLASSES + NUM_KEYPOINTS * 3, 8400)).T  # (8400, 79)
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
    
    # xywh → xyxy
    xy = boxes[:, :2] - boxes[:, 2:] / 2
    wh = boxes[:, :2] + boxes[:, 2:]
    boxes_xyxy = np.concatenate((xy, wh), axis=1)

    # NMS (간단한 버전)
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

# NMS용 IoU 계산 (NumPy)
def compute_iou_np(box1, boxes2):
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / union

# 5. 추론 함수
def infer(frame):
    orig_shape = frame.shape[:2]
    img_input = preprocess_frame(frame)
    cuda.memcpy_htod(d_input, img_input)
    context.execute_v2(bindings)
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    return postprocess(output, orig_shape)
