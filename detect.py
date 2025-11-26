import argparse
import os
import time
from pathlib import Path

# 기본 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO
from PIL import Image

# ---------------------- ColorThief 관련 함수 ----------------------

def extract_dominant_color(image, k=3):
    """
    주어진 이미지에서 주요 색상을 추출합니다.
    Args:
        image (numpy.ndarray): BGR 이미지
        k (int): 클러스터 수
    Returns:
        dominant_color (tuple): RGB 형식의 주요 색상
        dominant_percentage (float): 주요 색상의 비율
    """
    # 이미지 확대
    scale_percent = 200  # 200% 확대
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

    # 노이즈 제거
    blur = cv2.GaussianBlur(resized_img, (5, 5), 0)

    # 색상 공간 변환
    img_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    # K-평균 클러스터링 적용
    pixel_data = img_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixel_data)

    # 주요 색상 추출
    colors_centers = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    percentages = counts / counts.sum()

    # 가장 비율이 높은 색상 선택
    dominant_color = colors_centers[np.argmax(percentages)]
    dominant_percentage = np.max(percentages)

    return tuple(dominant_color), dominant_percentage


def create_color_image(color, size=(100, 100)):
    """
    지정된 색상으로 단색 이미지를 생성합니다.
    Args:
        color (tuple): RGB 형식의 색상
        size (tuple): 이미지 크기 (width, height)
    Returns:
        color_img (numpy.ndarray): 단색 이미지
    """
    color_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    color_img[:] = color  # 이미지를 지정된 색상으로 채우기
    return color_img


def process_detected_objects(save_dir, im0, detections, names, line_thickness=3):
    """
    감지된 객체들에 대해 색상을 추출하고 저장.
    Args:
        save_dir (Path): 저장할 디렉토리 경로
        im0 (numpy.ndarray): 원본 이미지
        detections (list): 감지된 객체들
        names (list): 클래스 이름 리스트
        line_thickness (int): 바운딩 박스 두께
    """
    # 각 클래스별 최고 신뢰도 및 해당 크롭된 이미지 저장을 위한 딕셔너리 초기화
    max_conf_per_class = {}  # {class_index: (confidence, cropped_image)}

    for det in detections:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                c = int(cls)  # 클래스 인덱스
                confidence = float(conf)
                # 너무 과도한 범위가 추출되면 정확한 색이 추출 안될 가능성이 있다고 판단함.
                # 따라서 크롭(섹션을 구분하는)박스의 크기를 의도적으로 줄임
                # 크롭 박스의 크기를 줄이기 위한 스케일 팩터 (예: 20% 줄임)
                scale_factor = 0.8  # 원하는 만큼 줄일 수 있음 (0.8은 20% 줄이는 의미)

                # xyxy 좌표를 조정하여 크기를 줄임
                x1, y1, x2, y2 = map(int, xyxy)  # 기존 좌표

                # 박스의 중심점 계산
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # 너비와 높이 계산
                width = (x2 - x1) * scale_factor
                height = (y2 - y1) * scale_factor

                # 새로운 좌표 계산 (이미지 경계를 벗어나지 않도록 조정)
                new_x1 = max(int(cx - width / 2), 0)
                new_y1 = max(int(cy - height / 2), 0)
                new_x2 = min(int(cx + width / 2), im0.shape[1])
                new_y2 = min(int(cy + height / 2), im0.shape[0])

                # 크롭된 이미지 얻기
                crop_img = im0[new_y1:new_y2, new_x1:new_x2].copy()

                # 클래스별 최대 신뢰도 업데이트
                if (c not in max_conf_per_class) or (confidence > max_conf_per_class[c][0]):
                    max_conf_per_class[c] = (confidence, crop_img)

    # 각 클래스별 최고 신뢰도의 크롭된 이미지 처리
    for c, (conf, crop_img) in max_conf_per_class.items():
        class_name = names[c]
        final_save_path = save_dir / f'best_{class_name}.jpg'
        cv2.imwrite(str(final_save_path), crop_img)
        print(f"클래스 '{class_name}'의 최고 신뢰도 크롭 이미지를 저장했습니다: {final_save_path}")


def extract_and_save_colors(save_dir):
    """
    저장된 이미지들에서 주요 색상을 추출하고 저장합니다.
    Args:
        save_dir (Path): 이미지가 저장된 디렉토리 경로
    """
    # 저장된 이미지들 순회
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)

        # 이미지 파일만 처리
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.endswith('_color.jpg'):
            print(f"{filename} 처리 중...")
            img = cv2.imread(file_path)
            if img is None:
                print(f"이미지를 불러올 수 없습니다: {file_path}")
                continue

            dominant_color, dominant_percentage = extract_dominant_color(img)

            if dominant_color is None:
                continue  # 이미지 로드 실패 시 건너뜀

            # 추출된 주요 색상을 파일로 저장 (파일명 + "_color.txt")
            color_filename = os.path.splitext(filename)[0] + '_color.txt'
            color_file_path = os.path.join(save_dir, color_filename)

            with open(color_file_path, 'w') as f:
                f.write(f'주요 색상 (RGB): {dominant_color}\n')
                f.write(f'비율: {dominant_percentage * 100:.2f}%\n')

            # 주요 색상 이미지를 저장 (파일명 + "_color.jpg")
            color_image = create_color_image(dominant_color)
            color_image_filename = os.path.splitext(filename)[0] + '_color.jpg'
            color_image_path = os.path.join(save_dir, color_image_filename)
            cv2.imwrite(color_image_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            print(f'{filename} 처리 완료: 주요 색상이 {color_file_path}와 {color_image_path}에 저장되었습니다.')


# ---------------------- 퍼스널 컬러 분석 관련 함수 ----------------------

# ---------------------- 퍼스널 컬러 분석 관련 함수 ----------------------

# 퍼스널 컬러 팔레트 정의 (가을 웜톤, 여름 쿨톤, 겨울 쿨톤, 봄 웜톤)
autumn_warm_palette = np.array([
    [61, 47, 47],    # #3d2f2f
    [66, 60, 45],    # #423c2d
    [37, 54, 51],    # #253634
    [41, 40, 48],    # #292830
    [111, 56, 38],   # #6F3826
])

winter_cool_palette = np.array([
    [191, 52, 52],    # #BF3434
    [217, 101, 35],   # #D96523
    [217, 143, 7],    # #D98F07
    [191, 179, 4],    # #BFB304
    [71, 166, 3],     # #47A603
])

summer_cool_palette = np.array([
    [176, 224, 230],  # #B0E0E6
    [123, 104, 238],  # #7B68EE
    [173, 216, 230],  # #ADD8E6
    [176, 196, 222],  # #B0C4DE
    [95, 158, 160],    # #5F9EA0
])

spring_warm_palette = np.array([
    [255, 182, 193],   # #FFB6C1
    [255, 160, 122],   # #FFA07A
    [250, 128, 114],   # #FA8072
    [255, 218, 185],   # #FFDAB9
    [255, 228, 196],   # #FFE4C4
])


def analyze_personal_color(colors):
    """
    퍼스널 컬러를 분석하는 함수
    Args:
        colors (list of tuple): 주요 색상 리스트 (RGB 형식)
    Returns:
        personal_color (str): 분석된 퍼스널 컬러
    """
    autumn_score = 0
    winter_score = 0
    summer_score = 0
    spring_score = 0

    for color in colors:
        color_array = np.array(color)

        # 각 팔레트와의 색상 차이를 계산
        diff_autumn = np.linalg.norm(autumn_warm_palette - color_array, axis=1)
        diff_winter = np.linalg.norm(winter_cool_palette - color_array, axis=1)
        diff_summer = np.linalg.norm(summer_cool_palette - color_array, axis=1)
        diff_spring = np.linalg.norm(spring_warm_palette - color_array, axis=1)

        # 가장 가까운 팔레트와의 차이로 점수 부여
        if min(diff_autumn) != 0:
            autumn_score += 1 / min(diff_autumn)
        else:
            autumn_score += 1

        if min(diff_winter) != 0:
            winter_score += 1 / min(diff_winter)
        else:
            winter_score += 1

        if min(diff_summer) != 0:
            summer_score += 1 / min(diff_summer)
        else:
            summer_score += 1

        if min(diff_spring) != 0:
            spring_score += 1 / min(diff_spring)
        else:
            spring_score += 1

    # 가장 높은 점수를 기준으로 퍼스널 컬러 진단
    scores = {
        'Autumn Warm Tone': autumn_score,
        'Winter Cool Tone': winter_score,
        'Summer Cool Tone': summer_score,
        'Spring Warm Tone': spring_score,
    }

    # 점수가 가장 높은 퍼스널 컬러 반환
    return max(scores, key=scores.get)


def process_best_skin_image(folder_path):
    """
    best_skin_color.jpg만 분석하는 함수
    Args:
        folder_path (str): 이미지가 저장된 폴더 경로
    """
    filename = 'best_skin_color.jpg'
    file_path = os.path.join(folder_path, filename)

    if os.path.exists(file_path):
        image = Image.open(file_path)

        # 이미지에서 주요 색상 추출
        dominant_color, _ = extract_dominant_color(np.array(image))
        dominant_colors = [dominant_color]  # 단일 색상인 경우 리스트로 감싸기

        # 퍼스널 컬러 분석
        personal_color = analyze_personal_color(dominant_colors)

        print(f"File: {filename}, Personal Color: {personal_color}")
    else:
        print(f"File {filename} not found in the directory.")



# ---------------------- YOLOv5과 YOLOv11 Detection 및 메인 함수 ----------------------

def get_next_save_dir(project, name):
    """
    결과를 저장할 다음 폴더 이름을 결정합니다.
    Args:
        project (str): 프로젝트 디렉토리 경로
        name (str): 기본 폴더 이름
    Returns:
        Path: 다음에 생성할 폴더의 경로
    """
    save_dir = Path(project) / name
    if not save_dir.exists():
        return save_dir
    else:
        i = 1
        while True:
            new_save_dir = Path(project) / f"{name}{i}"
            if not new_save_dir.exists():
                return new_save_dir
            i += 1


def run(
        weights1="yolo5best.pt",  # 첫 번째 모델 경로 
        weights2="yolo11best.pt",  # 두 번째 모델 경로 
        source="0",  # 파일/디렉토리/URL/glob/screen/webcam
        imgsz=(416, 416),  # 추론 이미지 크기 (높이, 너비)
        conf_thres=0.5,  # 신뢰도 임계값
        iou_thres=0.45,  # NMS IoU 임계값
        max_det=1000,  # 이미지당 최대 검출 수
        device="",  # CUDA 디바이스 또는 CPU
        view_img=True,  # 결과를 화면에 표시할지 여부
        save_txt=False,  # 결과를 텍스트 파일로 저장할지 여부
        save_format=0,  # 박스 좌표를 YOLO 또는 Pascal-VOC 형식으로 저장
        save_csv=False,  # 결과를 CSV로 저장할지 여부
        save_conf=False,  # 신뢰도 점수를 저장할지 여부
        save_crop=False,  # 검출된 객체를 크롭하여 저장할지 여부
        nosave=False,  # 이미지/비디오를 저장하지 않을지 여부
        classes=None,  # 특정 클래스만 필터링할지 여부
        agnostic_nms=False,  # 클래스에 무관한 NMS 적용 여부
        augment=False,  # 추론 시 데이터 증강 적용 여부
        visualize=False,  # 특징 맵 시각화 여부
        update=False,  # 모든 모델 업데이트 여부
        exist_ok=False,  # 기존 폴더에 덮어쓸지 여부
        line_thickness=3,  # 바운딩 박스 두께
        hide_labels=False,  # 레이블 숨길지 여부
        hide_conf=False,  # 신뢰도 점수 숨길지 여부
        half=False,  # FP16 반정밀도 추론 사용 여부
        dnn=False,  # ONNX 추론 시 OpenCV DNN 사용 여부
        vid_stride=1,  # 비디오 프레임 간격
        #아래 코드부분 경로 수정 필요
        project=r"path",  # 저장할 프로젝트 디렉토리
        name="result",  # 저장할 결과 디렉토리 이름
):
    """
    두 개의 YOLO 모델을 사용하여 객체 감지를 수행하고, 신뢰도가 높은 부분만 크롭하여 저장
    """
    # 디바이스 설정: 지정되지 않았으면 'cpu'로 설정
    device = device if device else 'cpu'
    print(f"Using device: {device}")

    # YOLOv5 모델 로드 via torch hub
    print("Loading YOLOv5 model...")
    try:
        model5 = torch.hub.load('ultralytics/yolov5', 'custom', path=weights1, force_reload=True)
        model5.to(device).eval()
    except Exception as e:
        print(f"YOLOv5 모델 로드 중 오류 발생: {e}")
        return

    # YOLOv11 모델 로드 via ultralytics
    print("Loading YOLOv11 model...")
    try:
        model11 = YOLO(weights2)
        model11.to(device)
    except Exception as e:
        print(f"YOLOv11 모델 로드 중 오류 발생: {e}")
        return

    # 저장 디렉토리 설정
    save_dir = get_next_save_dir(project, name) if not exist_ok else Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"Results will be saved to: {save_dir}")

    # 입력 소스 로드
    # 여기서는 웹캠 또는 비디오 파일을 처리하는 예시입니다.
    try:
        source_input = int(source) if source.isnumeric() else str(source)
    except ValueError:
        source_input = str(source)
    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    start_time = time.time()
    duration = 5  # 캡처 지속 시간 (초)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 불러올 수 없습니다.")
            break

        current_time = time.time()
        if current_time - start_time > duration:
            print("설정된 시간이 경과하여 캡처를 종료합니다.")
            break  # 설정된 시간이 지나면 루프를 종료

        # YOLOv5 추론
        results5 = model5(frame)
        detections5 = results5.xyxy[0].cpu().numpy()  # numpy array: [x1, y1, x2, y2, conf, cls]

        # YOLOv11 추론
        results11 = model11(frame)
        detections11 = []
        if results11:
            boxes11 = results11[0].boxes
            if boxes11:
                # YOLOv11의 boxes.xyxy, boxes.conf, boxes.cls를 이용하여 detections11을 구성
                xyxy = boxes11.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes11.conf.cpu().numpy()  # [conf]
                cls = boxes11.cls.cpu().numpy()    # [cls]
                for i in range(len(xyxy)):
                    detections11.append([
                        xyxy[i][0],
                        xyxy[i][1],
                        xyxy[i][2],
                        xyxy[i][3],
                        conf[i],
                        cls[i]
                    ])

        # 감지된 객체들을 리스트로 변환
        list_detections5 = []
        if detections5.size > 0:
            for det in detections5:
                x1, y1, x2, y2, conf, cls = det
                list_detections5.append([
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    cls
                ])

        list_detections11 = detections11  # 이미 리스트 형태로 변환됨

        # 두 모델의 결과 병합
        combined_detections = list_detections5 + list_detections11

        # 중복 제거 및 신뢰도 높은 감지만 유지
        # 동일 클래스 및 겹치는 박스 중 신뢰도 높은 것만 선택
        final_detections = []

        # 정렬: 높은 신뢰도 순으로 정렬
        combined_detections = sorted(combined_detections, key=lambda x: x[4], reverse=True)

        for det in combined_detections:
            x1, y1, x2, y2, conf, cls = det
            # 현재 감지가 다른 final_detections과 겹치는지 확인
            overlap = False
            for final_det in final_detections:
                fx1, fy1, fx2, fy2, fconf, fcls = final_det
                if cls == fcls:
                    # IoU 계산
                    inter_x1 = max(x1, fx1)
                    inter_y1 = max(y1, fy1)
                    inter_x2 = min(x2, fx2)
                    inter_y2 = min(y2, fy2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (fx2 - fx1) * (fy2 - fy1)
                        iou = intersection / (area1 + area2 - intersection)

                        if iou > 0.5:
                            overlap = True
                            break
            if not overlap:
                final_detections.append(det)

        # 최종 감지된 객체 처리 및 크롭
        if final_detections:
            process_detected_objects(save_dir, frame, [final_detections], model5.names, line_thickness)

    cap.release()

    # YOLO 감지가 완료된 후, 저장된 이미지들에 대해 ColorThief 기능을 적용
    extract_and_save_colors(save_dir)

    # 퍼스널 컬러 분석 실행
    process_best_skin_image(str(save_dir))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights1", type=str, default="yolo5best.pt", help="첫 번째 모델 경로 (YOLOv5 via torch hub)")
    parser.add_argument("--weights2", type=str, default="yolo11best.pt", help="두 번째 모델 경로 (YOLOv11 via ultralytics)")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/webcam")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[416], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", default=True, help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="YOLO (0) or Pascal-VOC (1) format for saving boxes coordinates when save-txt is True",
    )

def run(
        weights1="yolo5best.pt",  # 첫 번째 모델 경로 
        weights2="yolo11best.pt",  # 두 번째 모델 경로 
        source="0",  # 파일/디렉토리/URL/glob/screen/webcam
        imgsz=(416, 416),  # 추론 이미지 크기 (높이, 너비)
        conf_thres=0.5,  # 신뢰도 임계값
        iou_thres=0.45,  # NMS IoU 임계값
        max_det=1000,  # 이미지당 최대 검출 수
        device="",  # CUDA 디바이스 또는 CPU
        view_img=True,  # 결과를 화면에 표시할지 여부
        save_txt=False,  # 결과를 텍스트 파일로 저장할지 여부
        save_format=0,  # 박스 좌표를 YOLO 또는 Pascal-VOC 형식으로 저장
        save_csv=False,  # 결과를 CSV로 저장할지 여부
        save_conf=False,  # 신뢰도 점수를 저장할지 여부
        save_crop=False,  # 검출된 객체를 크롭하여 저장할지 여부
        nosave=False,  # 이미지/비디오를 저장하지 않을지 여부
        classes=None,  # 특정 클래스만 필터링할지 여부
        agnostic_nms=False,  # 클래스에 무관한 NMS 적용 여부
        augment=False,  # 추론 시 데이터 증강 적용 여부
        visualize=False,  # 특징 맵 시각화 여부
        update=False,  # 모든 모델 업데이트 여부
        exist_ok=False,  # 기존 폴더에 덮어쓸지 여부
        line_thickness=3,  # 바운딩 박스 두께
        hide_labels=False,  # 레이블 숨길지 여부
        hide_conf=False,  # 신뢰도 점수 숨길지 여부
        half=False,  # FP16 반정밀도 추론 사용 여부
        dnn=False,  # ONNX 추론 시 OpenCV DNN 사용 여부
        vid_stride=1,  # 비디오 프레임 간격
        #아래 코드부분 경로 수정 필요
        project=r"path",  # 저장할 프로젝트 디렉토리
        name="result",  # 저장할 결과 디렉토리 이름
):
    """
    두 개의 YOLO 모델을 사용하여 객체 감지를 수행하고, 신뢰도가 높은 부분만 크롭하여 저장
    """
    # 디바이스 설정: 지정되지 않았으면 'cpu'로 설정
    device = device if device else 'cpu'
    print(f"Using device: {device}")

    # YOLOv5 모델 로드 via torch hub
    print("Loading YOLOv5 model...")
    try:
        model5 = torch.hub.load('ultralytics/yolov5', 'custom', path=weights1, force_reload=True)
        model5.to(device).eval()
    except Exception as e:
        print(f"YOLOv5 모델 로드 중 오류 발생: {e}")
        return

    # YOLOv11 모델 로드 via ultralytics
    print("Loading YOLOv11 model...")
    try:
        model11 = YOLO(weights2)
        model11.to(device)
    except Exception as e:
        print(f"YOLOv11 모델 로드 중 오류 발생: {e}")
        return

    # 저장 디렉토리 설정
    save_dir = get_next_save_dir(project, name) if not exist_ok else Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"Results will be saved to: {save_dir}")

    # 입력 소스 로드
    # 여기서는 웹캠 또는 비디오 파일을 처리하는 예시입니다.
    try:
        source_input = int(source) if source.isnumeric() else str(source)
    except ValueError:
        source_input = str(source)
    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    start_time = time.time()
    duration = 5  # 캡처 지속 시간 (초)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 불러올 수 없습니다.")
            break

        current_time = time.time()
        if current_time - start_time > duration:
            print("설정된 시간이 경과하여 캡처를 종료합니다.")
            break  # 설정된 시간이 지나면 루프를 종료

        # YOLOv5 추론
        results5 = model5(frame)
        detections5 = results5.xyxy[0].cpu().numpy()  # numpy array: [x1, y1, x2, y2, conf, cls]

        # YOLOv11 추론
        results11 = model11(frame)
        detections11 = []
        if results11:
            boxes11 = results11[0].boxes
            if boxes11:
                # YOLOv11의 boxes.xyxy, boxes.conf, boxes.cls를 이용하여 detections11을 구성
                xyxy = boxes11.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes11.conf.cpu().numpy()  # [conf]
                cls = boxes11.cls.cpu().numpy()    # [cls]
                for i in range(len(xyxy)):
                    detections11.append([
                        xyxy[i][0],
                        xyxy[i][1],
                        xyxy[i][2],
                        xyxy[i][3],
                        conf[i],
                        cls[i]
                    ])

        # 감지된 객체들을 리스트로 변환
        list_detections5 = []
        if detections5.size > 0:
            for det in detections5:
                x1, y1, x2, y2, conf, cls = det
                list_detections5.append([
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    cls
                ])

        list_detections11 = detections11  # 이미 리스트 형태로 변환됨

        # 두 모델의 결과 병합
        combined_detections = list_detections5 + list_detections11

        # 중복 제거 및 신뢰도 높은 감지만 유지
        # 동일 클래스 및 겹치는 박스 중 신뢰도 높은 것만 선택
        final_detections = []

        # 정렬: 높은 신뢰도 순으로 정렬
        combined_detections = sorted(combined_detections, key=lambda x: x[4], reverse=True)

        for det in combined_detections:
            x1, y1, x2, y2, conf, cls = det
            # 현재 감지가 다른 final_detections과 겹치는지 확인
            overlap = False
            for final_det in final_detections:
                fx1, fy1, fx2, fy2, fconf, fcls = final_det
                if cls == fcls:
                    # IoU 계산
                    inter_x1 = max(x1, fx1)
                    inter_y1 = max(y1, fy1)
                    inter_x2 = min(x2, fx2)
                    inter_y2 = min(y2, fy2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (fx2 - fx1) * (fy2 - fy1)
                        iou = intersection / (area1 + area2 - intersection)

                        if iou > 0.5:
                            overlap = True
                            break
            if not overlap:
                final_detections.append(det)

        # 최종 감지된 객체 처리 및 크롭
        if final_detections:
            process_detected_objects(save_dir, frame, [final_detections], model5.names, line_thickness)

    cap.release()

    # YOLO 감지가 완료된 후, 저장된 이미지들에 대해 ColorThief 기능을 적용
    extract_and_save_colors(save_dir)

    # 퍼스널 컬러 분석 실행
    process_best_skin_image(str(save_dir))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights1", type=str, default="yolo5best.pt", help="첫 번째 모델 경로 (YOLOv5 via torch hub)")
    parser.add_argument("--weights2", type=str, default="yolo11best.pt", help="두 번째 모델 경로 (YOLOv11 via ultralytics)")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/webcam")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[416], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", default=True, help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="YOLO (0) or Pascal-VOC (1) format for saving boxes coordinates when save-txt is True",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    
    #민선홍 아래 파일 경로 수정 필요
    parser.add_argument("--project", default=BASE_DIR, help="save results to project/name")
    parser.add_argument("--name", default="result", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    # 이미지 크기가 하나의 값이면 (예: 416), 이를 (416, 416)으로 변환
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    return opt


def main(opt):
    run(
        weights1=opt.weights1,
        weights2=opt.weights2,
        source=opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        device=opt.device,
        view_img=opt.view_img,
        save_txt=opt.save_txt,
        save_format=opt.save_format,
        save_csv=opt.save_csv,
        save_conf=opt.save_conf,
        save_crop=opt.save_crop,
        nosave=opt.nosave,
        classes=opt.classes,
        agnostic_nms=opt.agnostic_nms,
        augment=opt.augment,
        visualize=opt.visualize,
        update=opt.update,
        exist_ok=opt.exist_ok,
        line_thickness=opt.line_thickness,
        hide_labels=opt.hide_labels,
        hide_conf=opt.hide_conf,
        half=opt.half,
        dnn=opt.dnn,
        vid_stride=opt.vid_stride,
        project=opt.project,
        name=opt.name,
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# 이미지 폴더 경로 (수정하기)
folder_path = os.path.join(BASE_DIR, 'result')

# 결과 출력
# file_results, final_color, average_distances = diagnose_personal_color_from_folder(folder_path)

# print("각 파일의 퍼스널 컬러 진단 결과:")
# for file, (best_match, distances) in file_results.items():
#     print(f"\n파일: {file}")
#     print("퍼스널 컬러별 유클리드 거리 값:")
#     for palette, dist in distances.items():
#         print(
#             f"{palette} - RGB: {dist['RGB']:.2f}, HSV: {dist['HSV']:.2f}, HEX: {dist['HEX']:.2f}, Total: {dist['Total']:.2f}")
#     print(f"최적 퍼스널 컬러: {best_match}, 최솟값 거리 합: {distances[best_match]['Total']:.2f}")

# 최종 결과 출력: 전체 파일에 대한 최종 퍼스널 컬러 진단 결과
# print("\n전체 파일에 대한 최종 퍼스널 컬러 진단 결과:")
# print(f"가장 유사한 최종 퍼스널 컬러: {final_color}")