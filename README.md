<h1>🖼️ Personal Color Detection & Cosmetic Recommendation System</h1>

본 프로젝트는 YOLO 기반 얼굴 부위 탐지, 랜드마크 기반 색상 추출, 퍼스널 컬러 분석,
그리고 화장품 추천 기능을 포함하는 통합 뷰티 분석 시스템입니다.

<div align="center"> 프로젝트 전체 파이프라인은 아래와 같습니다.<br><br> <img width="209" height="315" alt="Pipeline" src="https://github.com/user-attachments/assets/c55733c5-9872-458f-b55a-49fd964bbf45" /> </div>
<h3>📥 YOLO 학습 데이터 다운로드</h3>

YOLO 모델 학습에 사용된 데이터는 아래 링크에서 다운로드할 수 있습니다.

https://drive.google.com/drive/folders/1k4H-39QyItacNX3GIBIU0mOp6ciYkAE7?usp=sharing

<h3>📄 논문 및 발표 정보</h3>

본 프로젝트는 2024년도 동계 KICS 포스터 세션에서 발표되었습니다.
연구 논문은 아래에서 확인할 수 있습니다.

https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12132276

저는 해당 연구에서

YOLO 학습,

얼굴 랜드마크 기반 색상 추출,

퍼스널 컬러 분석 모듈 개발
부분을 담당하였습니다.


<h3>🎯 YOLO 앙상블 기반 퍼스널 컬러 분석</h3>

본 프로젝트는 YOLOv5와 YOLOv11 모델을 조합하여 얼굴의 피부, 눈썹, 입술과 같은 주요 부위를 탐지하며,
모델 간 성능의 상호보완성을 활용한 최대 신뢰도 선택(Maximum Confidence Selection) 기법을 적용합니다.
또한 IoU 기반 중복 제거를 통해 탐지 결과의 정확도를 높였습니다.

<h3>✔ YOLO 모델 성능 비교 (Confusion Matrix 기반)</h3>
<div align="center"> <img width="1238" height="445" alt="Poster" src="https://github.com/user-attachments/assets/121bd055-ce1e-4c30-9c4c-2d7dacca4345" /> </div>
두 모델의 특성은 다음과 같습니다:


<div align="center">


| 클래스           | YOLOv5 정확도 | YOLOv11 정확도 |
| ------------- | ---------- | ----------- |
| Upper lip     | **0.93**   | 0.02        |
| Lower lip     | **0.76**   | 0.03        |
| Skin          | 0.45       | **0.94**    |
| Left eyebrow  | 0.37       | **0.62**    |
| Right eyebrow | 0.40       | **0.83**    |

</div>
YOLOv5는 lip(입술) 영역에서 높은 정확도를 보입니다.

YOLOv11은 skin·eyebrow(피부·눈썹) 영역에서 높은 탐지 성능을 보입니다.

두 모델은 서로 다른 영역에서 강점을 보이므로 앙상블 시 성능이 크게 향상됩니다.

<h3>✔ Maximum Confidence Selection 기법</h3>

여러 모델이 동일한 객체를 탐지하면,
confidence가 가장 높은 결과만 선택하여 최종 박스로 사용합니다.

이는 색상 기반 퍼스널 컬러 분석에서 오차를 줄이고,
부위별 가장 정확한 추출 색상을 확보하는 데 효과적입니다.

<h3>✔ IoU 기반 중복 제거</h3>

두 모델이 동일 객체를 탐지할 경우 IoU ≥ 0.5 기준을 적용하여
신뢰도가 낮은 박스를 제거합니다.

이를 통해:

중복 탐지를 방지하고

주요 부위의 정확한 크롭 이미지를 확보합니다.

<h3>✔ 주요 색상 추출 → 퍼스널 컬러 판단</h3>

피부 · 눈썹 · 입술를 크롭

RGB 색상 기반 K-Means 클러스터링(K=3)

클러스터 중심값을 대표 색상으로 선정

팔레트(봄·여름·가을·겨울)와 RGB 거리 계산

가장 가까운 팔레트를 최종 퍼스널 컬러로 판단

이렇게 분석된 퍼스널 컬러는 이후 화장품 추천 시스템에서 활용됩니다.

<h3>🎨 퍼스널 컬러 설명 화면</h3>
<img width="1917" height="851" alt="PersonalColor" src="https://github.com/user-attachments/assets/600ef9c9-0a70-4090-b529-57dbacc32e16" />

<h3>📊 분석 결과 화면</h3>
<div align="center"> <img width="421" height="418" alt="Result" src="https://github.com/user-attachments/assets/77b210d8-d31e-4e23-b69c-4439263e8d3b" /> </div>

<h3>💄 화장품 추천 페이지</h3>
<div align="center"> <img width="1915" height="895" alt="CosmeticPage" src="https://github.com/user-attachments/assets/4bee1dae-5119-4a25-ae7e-d3d10b005f21" /> </div>

<h3>📚 자세한 내용</h3>

프로젝트의 알고리즘, 실험 결과, 시스템 구조 등 자세한 내용은 논문에서 확인할 수 있습니다.
