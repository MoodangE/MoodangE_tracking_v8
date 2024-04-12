# MoodangE_tracking_v8

가천대학교 교내 에코버스 무당이의 경로 탐색과 정류장 사람 혼잡도를 파악을 위한 프로젝트 입니다.

기존 [MoodangE_tracking](https://github.com/MoodangE/MoodangE_tracking)은 [YOLOv5](https://github.com/ultralytics/yolov5)를
사용했으나, 새로운 [YOLOv8](https://github.com/ultralytics/ultralytics) 버전을 적용하여 리뉴얼하였습니다.

---

# Models

### Person Detection

기본으로 `yolov8n-person.pt`가 지정되어있습니다. <br/>
사람 탐지를 위해 `yolov8n-person.pt`, `yolov8n-face.pt` 를 선택해서 사용할 수 있습니다. <br/>
해당 모델들은 [YOLOv8-face](https://github.com/akanametov/yolov8-face)에 있는 학습되어 있는 모델을 사용하였습니다.

---

# Install

해당 프로젝트는 아나콘다 환경에서 Python 3.10 기반입니다.

### Mac 버전

Mac에서 사용 가능한 Pytorch를 갖춘 Conda env를 import합니다.

```bash
conda env create -f MoodangE_tracking_v8_mac.yml
```

### Window 버전

Window에서 사용 가능한 Pytroch를 기반으로 하여 Conda env를 import합니다.

```bash
conda env create -f MoodangE_tracking_v8_window.yml
```

# Usage

### 혼잡도 파악 (단일 영상 적용을 권장)
<img width="100%" src="https://github.com/MoodangE/MoodangE_tracking/assets/71388566/13b96a2e-7d2a-45fb-b727-01c98217e024" alt="person_congestion exmaple result"></a>
 [사용 영상 출처: City Street on Christmas](https://www.pexels.com/video/city-street-on-christmas-6057901/)
```bash
python person_congestion.py
```

파라미터를 통해 혼잡도 파악 시 적절한 값을 설정할 수 있습니다.
<details>
<summary>Parameters</summary>

- **YOLOv8** 관련
    - `-weight`: 사용할 모델의 경로를 지정합니다. 기본값은`'models/yolov8n-person.pt'`입니다. 이는 사람을 감지하기 위한 사전 학습된 모델을 가리킵니다.
    - `-source`: 감지를 수행할 대상의 경로입니다. 파일, 디렉토리, URL, 글로브 패턴을 지정할 수 있으며, 웹캠을 사용하기 위해서는`0`을 입력합니다.
      기본값은`'ultralytics/assets/bus.jpg'`입니다.
    - `-no-save`: 이 옵션을 사용하면, 감지된 이미지나 비디오를 저장하지 않습니다.
    - `-device`: 사용할 컴퓨팅 장치를 지정합니다. 예를 들어, CUDA 기반 GPU를 사용하려면`0`또는`0,1,2,3`과 같이 지정하고, CPU를 사용하려면`cpu`를 입력합니다. 기본값은 빈
      문자열이며, 이 경우 시스템이 가용한 장치를 자동으로 선택합니다. 
      - Mac에서 `mps` 사용을 원하는 경우 `--device mps` 설정이 필요하며, 사용을 원치 않는 경우 `--device cpu`를 반드시 설정해야 합니다. 
    - `-project`: 결과를 저장할 프로젝트의 이름입니다. 기본값은`'inference_person'`입니다.
    - `-name`: 결과를 저장할 폴더의 이름입니다. 기본값은`'exp'`입니다.
    - `-conf-thresh`: 객체 감지의 신뢰도 임계값입니다. 기본값은`0.25`입니다.
    - `-iou-thresh`: 비최대 억제(NMS)를 위한 IoU(Intersection over Union) 임계값입니다. 기본값은`0.45`입니다.
    - `-agnostic`: 이 옵션을 사용하면, 클래스에 구애받지 않고 확장된 추론을 수행합니다.
    - `-view-img`: 이 옵션을 사용하면, 결과를 화면에 표시합니다.
      <br/><br/>
- **SORT** 관련
    - `-sort-max-age`: 객체가 가리어지거나 n 프레임 동안 감지되지 않아도 추적을 유지하는 최대 프레임 수입니다. 기본값은`30`입니다.
    - `-sort-min-hits`: 추적을 시작하기 전에 감지해야 하는 객체의 최소 수입니다. 기본값은`2`입니다.
    - `-sort-iou-thresh`: 두 프레임 간의 객체 연관성을 결정하기 위한 IoU 임계값입니다. 기본값은`0.2`입니다.
      <br/><br/>
- **Person_congestion** 관련
    - `-blur`: 이 옵션을 사용하면, 감지된 객체의 경계 상자를 흐리게 처리합니다.
    - `-tracking`: 추적 경로를 시각화합니다. 이 옵션을 사용하면 객체의 이동 경로가 화면에 표시됩니다.
    - `-duration`: 이미지를 지정한 시간(초) 동안 처리합니다. 기본값은`5.0`입니다.

</details>

---
# 참고
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) : With No Modifications
- [abewley/sort](https://github.com/abewley/sort) :  With minor Modifications
- [akanametov/yolov8-face](https://github.com/akanametov/yolov8-face) : Use the provided learning model