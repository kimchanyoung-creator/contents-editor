# AI 자동 비디오 편집기

이미지, 영상, 음악을 업로드하면 자동으로 분석하여 뮤직비디오/영화 스타일의 결과물을 생성하는 Google Colab 기반 자동 편집 프로그램입니다.

## 주요 기능

### 미디어 분석
- **얼굴 감지**: MediaPipe를 이용한 정확한 얼굴 위치 감지
- **비트 분석**: Librosa를 이용한 음악 템포, 비트, 강한 비트 분석
- **구간 분석**: 음악의 에너지 변화를 분석하여 자동 구간 분할

### 영화적 카메라 기법
| 기법 | 설명 |
|------|------|
| ZOOM_IN | 와이드에서 클로즈업으로 줌인 |
| ZOOM_OUT | 클로즈업에서 와이드로 줌아웃 |
| PAN_LEFT/RIGHT | 좌/우 패닝 |
| TILT_UP/DOWN | 상/하 틸트 |
| DOLLY_IN/OUT | 부드러운 돌리 인/아웃 |
| SHAKE | 핸드헬드 카메라 효과 |
| KEN_BURNS | 사진에 동적 움직임 부여 |
| FACE_TRACK | 얼굴 중심 자동 줌 |

### 트랜지션 효과
- **CUT**: 즉시 전환
- **CROSSFADE**: 크로스페이드
- **FADE**: 블랙을 통한 페이드
- **FLASH**: 화이트아웃 플래시
- **WIPE**: 좌/우 와이프
- **ZOOM_TRANSITION**: 줌 트랜지션

### 지능형 편집
- 음악 비트에 맞춘 자동 클립 길이 조절
- 강한 비트에서 임팩트 있는 전환
- 얼굴이 감지되면 자동으로 얼굴 중심 줌인/줌아웃
- 미디어 특성에 따른 카메라 기법 자동 선택

## 지원 파일 형식

| 종류 | 확장자 |
|------|--------|
| 이미지 | jpg, jpeg, png, bmp, webp |
| 비디오 | mp4, avi, mov, mkv, webm |
| 오디오 | mp3, wav, ogg, m4a, flac |

## 사용 방법

### 1. Google Colab에서 열기

`auto_video_editor.ipynb` 파일을 Google Colab에서 엽니다.

### 2. 라이브러리 설치

첫 번째 셀을 실행하여 필요한 라이브러리를 설치합니다.

### 3. 파일 업로드 및 실행

마지막 실행 셀을 실행하면:
1. 파일 업로드 다이얼로그가 나타납니다
2. 이미지/비디오와 음악 파일을 업로드합니다
3. 자동으로 분석 및 편집이 진행됩니다
4. 완성된 비디오가 자동으로 다운로드됩니다

## 설정 커스터마이징

```python
# 출력 해상도
Config.OUTPUT_WIDTH = 1920   # 4K: 3840
Config.OUTPUT_HEIGHT = 1080  # 4K: 2160

# 클립 길이
Config.MIN_CLIP_DURATION = 1.5  # 최소 클립 길이 (초)
Config.MAX_CLIP_DURATION = 8.0  # 최대 클립 길이 (초)

# 줌 강도
Config.ZOOM_INTENSITY = 1.3      # 일반 줌 (1.0 = 변화없음)
Config.FACE_ZOOM_INTENSITY = 1.5  # 얼굴 줌

# 트랜지션 길이
Config.TRANSITION_DURATION = 0.5  # 초
```

## 기술 스택

- **Python 3.x**
- **MoviePy**: 비디오 편집 및 렌더링
- **OpenCV**: 이미지/비디오 처리
- **MediaPipe**: 얼굴 감지
- **Librosa**: 음악 분석
- **NumPy/SciPy**: 수치 계산

## 파일 구조

```
contents-editor/
├── auto_video_editor.ipynb  # 메인 Colab 노트북
└── README.md                # 프로젝트 설명
```

## 라이선스

MIT License
