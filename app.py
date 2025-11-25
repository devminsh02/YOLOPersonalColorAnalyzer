import norm
import pandas as pd
from flask import Flask, render_template, Response, redirect, url_for, request, jsonify,send_from_directory# 웹앱 개발에 필요한 기능을 Flask 모듈에서 가져오는 코드
import cv2        # OpenCV 라이브러리로 웹캠 영상을 처리
import os         # 파일 경로 및 폴더 처리를 위한 모듈
import shutil     # 파일 및 폴더 삭제, 복사를 위한 모듈
import subprocess # 외부 프로그램을 실행하기 위한 모듈
import numpy as np
from numpy.linalg import norm

from numpy import array

app = Flask(__name__) # Flask 애플리케이션을 생성
app.url_map.strict_slashes = False

# 초기화 코드 추가
result_file_name = ''
personal_color_result = ''

# 메인 페이지
@app.route('/')
def main():
    # 메인 페이지 HTML 파일을 렌더링하여 반환
    return render_template('main.html')

# 진단 페이지
@app.route('/diagnose')
def diagnose():
    # 진단 페이지 HTML 파일을 렌더링하여 반환
    return render_template('diagnose.html')

# 퍼스널 컬러 정보 페이지
@app.route('/color-info')
def color_info():
    return render_template('color_info.html')


# 웹캠 영상을 스트리밍하는 함수
def gen_frames():
    # OpenCV를 사용하여 기본 웹캠을 연결 (0번 인덱스의 카메라)
    # 0은 기본적으로 연결된 첫 번째 카메라를 의미 (노트북에 내장된 웹캠 등)
    camera = cv2.VideoCapture(0)

    # 무한 루프를 사용하여 웹캠에서 계속해서 영상을 읽어옴
    while True:
        success, frame = camera.read()  # 웹캠에서 프레임(이미지)을 읽어옴
        if not success:  # 만약 웹캠에서 프레임을 읽어오지 못하면,
            break  # 루프 종료 (ex. 카메라가 연결되지 않은 경우)
        else:
            # 읽어온 프레임을 JPEG 이미지 형식으로 인코딩
            # 웹캠에서 읽어온 이미지를 웹 브라우저에 전송하기 위해서
            # 일반 이미지 대신 압축된 JPEG 이미지로 변환
            ret, buffer = cv2.imencode('.jpg', frame)

            # 인코딩된 이미지를 바이트 형식으로 변환
            # Why? 바이트 형식이 되어야만 데이터를 HTTP 응답으로 전송 가능
            frame = buffer.tobytes()

            # HTTP 응답 형식으로 프레임을 하나씩 전송
            # 각 프레임은 MIME 형식(이미지 데이터를 포함하는 HTTP 형식)으로 전달
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # --frame: 새로운 프레임을 시작하는 구분자
            # Content-Type: image/jpeg: 응답으로 전송되는 데이터는 JPEG 이미지임을 표시
            # frame + b'\r\n': 실제 이미지 데이터를 전송하고 끝에 개행 문자를 추가


# 웹캠 영상을 스트리밍하는 경로 설정 (브라우저에서 웹캠 영상을 볼 수 있는 URL 경로)
@app.route('/video_feed')
def video_feed():
    # video_feed => 경로로 접근했을 때 웹캠 영상을 HTTP 응답으로 전송

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # gen_frames => 함수로부터 프레임을 계속 가져와서 스트리밍 형식으로 응답
    # Response => Flask의 Response 객체를 사용해 HTTP 응답을 보냄
    # gen_frames() => 웹캠에서 실시간으로 생성된 프레임을 가져옴
    # mimetype => 응답의 콘텐츠 타입을 정의 (여기서는 'multipart/x-mixed-replace'로 지정)
    # 'multipart/x-mixed-replace' => 웹 브라우저에서 지속적으로 이미지를 받는 형식
    # boundary=frame => 각 이미지 사이를 구분하는 경계(boundary)를 의미, 이를 통해 여러 프레임을 구분



# 진단 시작 버튼 클릭 시 YOLOv5 실행
@app.route('/run_yolo', methods=['POST'])
def run_yolo():
    global result_file_name, personal_color_result

    # 결과 디렉토리 설정
    result_directory = r'C:\Users\주윤호\Desktop\test'
    try:
        if os.path.exists(result_directory):
            shutil.rmtree(result_directory)
        os.makedirs(result_directory)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

    try:
        # detect.py 실행
        detect_script_path = r'detect.py'
        python_executable = r'C:\venvs\myproject\Scripts\python.exe'  # 가상 환경의 Python 경로

        if not os.path.exists(detect_script_path):
            print(f"Error: detect.py 파일이 {detect_script_path} 경로에 존재하지 않습니다.")
            return jsonify({'status': 'error', 'message': 'detect.py 파일이 경로에 존재하지 않습니다.'})

        result = subprocess.run(
            [python_executable, detect_script_path],
            capture_output=True, text=True, encoding='utf-8'
        )

        # 실행 결과 출력
        print("YOLO Detect Script Output:", result.stdout)
        print("YOLO Detect Script Error:", result.stderr)

        # 실행 결과에서 특정 정보를 파싱 (예: 결과 파일명과 퍼스널 컬러 정보)
        for line in result.stdout.splitlines():
            if 'File:' in line:
                result_file_name = line.split('File: ')[-1].strip()
            if 'Personal Color:' in line:
                personal_color_result = line.split('Personal Color: ')[-1].strip()

        return redirect(url_for('result'))

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 결과 페이지
@app.route('/result')
@app.route('/result/')
def result():
    return render_template('result.html', personal_color=personal_color_result)


# 파일 삭제 기능 (진단 시작 버튼을 누르면 정보 초기화)
@app.route('/delete', methods=['POST'])
def delete_files():
    folder_path = r'C:\Users\주윤호\Desktop\test'  # 삭제할 폴더 경로
    try:
        # 해당 폴더 내의 모든 파일 및 폴더 삭제
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)   # 파일 경로 생성
            if os.path.isfile(file_path):
                os.remove(file_path)       # 파일 삭제
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)   # 폴더 삭제
        return redirect(url_for('diagnose'))  # 삭제 후 진단 페이지 이동
    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('diagnose'))  # 에러 떠도 진단 페이지 이동

# 이미지 파일을 가져오기 위한 라우트 설정
@app.route('/result/<path:filename>')
def send_result(filename):
    # 바탕화면에 있는 test/result 폴더 경로를 설정하여 파일을 반환
    base_path = os.path.expanduser(r"C:\Users\주윤호\Desktop\test\result")
    return send_from_directory(base_path, filename)   # 파일을 폴더에서 사용자에게 전송

# 퍼스널 컬러 분석을 위한 HSV 립 팔레트 재정의
autumn_warm_palette = np.array([
    [30, 60, 50],  # 브라운 계열
    [25, 50, 40],  # 브릭 레드
    [35, 70, 60],  # 코랄 브라운
    [20, 55, 45],  # 딥 오렌지
    [40, 65, 55],  # 소프트 브라운
    [20, 70, 80],  # 테라코타
    [45, 75, 65], [50, 80, 70]  # 추가 색상
])

winter_cool_palette = np.array([
    [0, 0, 100],  # 퓨어 화이트
    [0, 0, 0],  # 제트 블랙
    [240, 100, 50],  # 로얄 블루
    [140, 100, 60],  # 에메랄드 그린
    [0, 100, 60],  # 루비 레드
    [210, 100, 70],  # 사파이어 블루
    [270, 100, 60],  # 아메시스트 퍼플
    [210, 5, 95],  # 아이스 그레이
    [0, 100, 100],  # 퓨어 레드
    [240, 100, 70],  # 코발트 블루
    [240, 70, 90],  # 딥 레드
    [260, 80, 100], # 버건디
    [220, 60, 80],  # 푸시아 핑크
    [310, 80, 40],  # 딥 플럼
    [250, 65, 85],  # 플럼
    [340, 70, 60],
    [230, 50, 75],   # 클래식 레드
    [350, 90, 70],   # 체리 레드 (Cherry Red)
    [270, 85, 95], [280, 90, 100]  # 추가 색상
])

summer_cool_palette = np.array([
    [210, 20, 95],  # 페일 블루
    [150, 25, 90],  # 소프트 민트
    [270, 15, 95],  # 라이트 라벤더
    [340, 20, 95],  # 파우더 핑크
    [210, 10, 85],  # 미스트 그레이
    [180, 30, 90],  # 라이트 아쿠아
    [60, 20, 95],   # 페일 옐로우
    [280, 25, 85],  # 소프트 퍼플
    [10, 30, 90],   # 라이트 코랄
    [200, 15, 95],   # 베이비 블루
    [200, 50, 80],  # 소프트 핑크
    [190, 45, 75],  # 라벤더 핑크
    [180, 40, 70],  # 라이트 로즈
    [210, 55, 85],  # 뮤트 푸시아
    [170, 35, 65],   # 페일 라일락
    [330, 20, 100],  # 라이트 핑크 (Light Pink)
    [340, 30, 90],   # 로즈 핑크 (Rose Pink)
    [270, 25, 95],   # 라벤더 (Lavender)
    [10, 20, 100],   # 소프트 코랄 (Soft Coral)
    [300, 20, 80],    # 모브 (Mauve)
    [160, 30, 60], [150, 25, 55]  # 추가 색상
])

spring_warm_palette = np.array([
    [60, 80, 90],   # 피치 코랄
    [70, 85, 95],   # 살구
    [55, 75, 85],   # 누드 피치
    [65, 83, 93],   # 라이트 오렌지
    [50, 78, 88],   # 소프트 코랄
    [75, 90, 98], [80, 95, 100]  # 추가 색상
])


# 주어진 HSV 값을 numpy 배열로 변환하는 함수
def parse_hsv(hsv_str):
    try:
        if isinstance(hsv_str, str):
            # 문자열에서 불필요한 괄호 제거하고, 쉼표로 나눈 뒤 각 값을 정수로 변환
            return np.array([int(x) for x in hsv_str.strip('()').split(',')])
    except Exception as e:
        # 오류 발생시 에러 메시지 출력
        print(f"Error parsing HSV: {hsv_str} - {e}")
    return np.array([0, 0, 0])  # 기본값 반환

# HSV 값을 정규화하는 함수
def normalize_hsv(hsv):
    h, s, v = hsv
    h = h % 360  # Hue(색상의 각도): 0~360 범위를 초과하지 않도록 조정
    s = min(max(s, 0), 100)  # Saturation(색의 채도): 0~100 범위로 제한
    v = min(max(v, 0), 100)  # Value(색의 밝기): 0~100 범위로 제한
    return np.array([h, s, v])

# HSV 값을 기반으로 화장품을 퍼스널 컬러로 분류하는 함수
def classify_personal_color(hsv):
    hsv = normalize_hsv(hsv)  # HSV 값 정규화
    print(f"Normalized HSV: {hsv}")

    # 입력 HSV값과 립 팔레트 HSV값 간의 차이
    diff_autumn = norm(autumn_warm_palette - hsv, axis=1)
    diff_winter = norm(winter_cool_palette - hsv, axis=1)
    diff_summer = norm(summer_cool_palette - hsv, axis=1)
    diff_spring = norm(spring_warm_palette - hsv, axis=1)

    print(f"Differences - Autumn: {min(diff_autumn)}, Winter: {min(diff_winter)}, Summer: {min(diff_summer)}, Spring: {min(diff_spring)}")

    # 거리 기반으로 점수 계산 (거리가 짧을수록 높은 점수 부여)
    scores = {
        'Autumn Warm Tone': 1 / (1 + min(diff_autumn)),
        'Winter Cool Tone': 1 / (1 + min(diff_winter)),
        'Summer Cool Tone': 1 / (1 + min(diff_summer)),
        'Spring Warm Tone': 1 / (1 + min(diff_spring)),
    }

    print(f"Scores: {scores}")
    # 점수가 가장 높은 퍼스널 컬러 반환
    return max(scores, key=scores.get)

# CSV 파일에서 데이터 읽기
try:
    data = pd.read_csv(
        r'C:\\projects\\myproject\\beauty_items\\colors.csv',
        engine='python',  # CSV 파일 파싱 엔진 (기본값 C, 대체 python)
        sep=',',  # 구분자 설정
        nrows=None,  # 모든 행 읽기
        skip_blank_lines=True  # 빈 줄 무시
    )
    print("CSV 데이터 로드 성공:")
    print(data.head())  # CSV 파일 데이터 확인

    # HSV 변환
    data['HSV'] = data['HSV'].apply(parse_hsv)

    # 퍼스널 컬러 분류
    data['color'] = data['HSV'].apply(classify_personal_color)

    # 퍼스널 컬러 분포 출력
    print("퍼스널 컬러 분포:")
    print(data['color'].value_counts())

except Exception as e:
    print(f"CSV 파일 로드 실패: {e}")
    data = pd.DataFrame()  # 빈 데이터프레임으로 초기화


# 주어진 퍼스널 컬러에 맞는 최상의 제품을 가져오는 함수
def get_best_products(color, full=False):

    # color열 유효성 검사 (color 열이 없으면 빈 데이터 프레임 반환)
    if 'color' not in data.columns:
        return pd.DataFrame()

    # color로 데이터 필터링 (공백 및 대소문자 X)
    filtered_data = data[data['color'].str.strip().str.lower() == color.strip().lower()]
    if filtered_data.empty:
        return pd.DataFrame()  # 비어있다면,,,빈 데이터프레임 반환

    # 퍼스널 컬러 팔레트 매핑
    palette_mapping = {
        'Autumn Warm Tone': autumn_warm_palette,
        'Winter Cool Tone': winter_cool_palette,
        'Summer Cool Tone': summer_cool_palette,
        'Spring Warm Tone': spring_warm_palette
    }
    target_palette = palette_mapping.get(color)
    if target_palette is None:
        return pd.DataFrame()

    # 4. 각 제품의 HSV 값과 타겟 팔레트 평균 HSV 간의 거리 계산
    filtered_data['distance'] = filtered_data['HSV'].apply(
        lambda x: norm(x - target_palette.mean(axis=0))
        # lambda 매개변수 : 반환 값 (, 이름 없는 함수라고 생각하면 됨.)
    )

    # full=True일 경우 모든 추천 제품 반환
    if full:
        recommended = filtered_data.sort_values(by='distance').reset_index(drop=True)
    # 아니면 상위 3개 제품 반환 (거리 계산을 통해 값을 정렬했기 때문에 가능)
    else:
        recommended = filtered_data.sort_values(by='distance').head(3).reset_index(drop=True)

    return recommended


# 추천 페이지 라우트
@app.route('/recommendations')
def recommendations():
    color = request.args.get('color')  # 요청된 퍼스널 컬러

    # 데이터가 비어 있는 경우 처리
    if data.empty:
        return render_template('recommendation.html', color=color, products=[])

    # 추천 제품 가져오기
    products = get_best_products(color).to_dict(orient='records')

    # 제품이 없는 경우 처리
    if not products:
        return render_template('recommendation.html', color=color, products=[], show_more_button=False)

    # 버튼 표시 여부 설정 (추천 제품이 3개를 초과하면 버튼 표시)
    show_more_button = len(data[data['color'] == color]) > 3

    # 추천 페이지 렌더링
    return render_template(
        'recommendation.html',
        color=color,
        products=products,
        show_more_button=show_more_button
    )


@app.route('/recommendations/more')
def more_recommendations():
    color = request.args.get('color')  # 요청된 퍼스널 컬러

    # 데이터가 비어 있는 경우 처리
    if data.empty:
        return render_template('recommendation.html', color=color, products=[])

    # 모든 추천 제품 가져오기
    all_products = get_best_products(color, full=True).to_dict(orient='records')

    return render_template('recommendation.html', color=color, products=all_products, show_more_button=False)

if __name__ == '__main__':
    app.run(debug=True)
