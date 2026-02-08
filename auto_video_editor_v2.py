#@title 🎬 AI 자동 비디오 편집기 v2.0
#@markdown ---
#@markdown ### ⚙️ 설정
#@markdown ---
output_format = "horizontal"  #@param ["horizontal", "vertical"] {allow-input: false}
#@markdown - **horizontal**: 가로형 (1920x1080) - YouTube, PC용
#@markdown - **vertical**: 세로형 (1080x1920) - 릴스, 틱톡, 쇼츠용
#@markdown ---
#@markdown ### 📁 실행하면 파일 업로드 창이 열립니다
#@markdown - 이미지 (jpg, png 등) + 음악 (mp3 등)
#@markdown - 또는 비디오 (mp4 등) + 음악 (mp3 등)
#@markdown - **음악 파일은 필수입니다!**
#@markdown ---

# ============================================================
# 1. 라이브러리 설치
# ============================================================
print("="*60)
print("📦 라이브러리 설치 중... (1-2분 소요)")
print("="*60)

import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'moviepy==1.0.3'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'librosa'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python-headless'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mediapipe'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scipy'])
subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], capture_output=True)

print("✅ 라이브러리 설치 완료!\n")

# ============================================================
# 2. 라이브러리 임포트
# ============================================================
print("📚 라이브러리 로딩 중...")

import os
import cv2
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import librosa
from scipy.signal import find_peaks
import mediapipe as mp

from google.colab import files
from IPython.display import HTML, display

from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, ColorClip,
    CompositeVideoClip, concatenate_videoclips
)

print("✅ 라이브러리 로드 완료!\n")

# ============================================================
# 3. 설정
# ============================================================
class Config:
    OUTPUT_FPS = 30
    MIN_CLIP_DURATION = 1.5
    MAX_CLIP_DURATION = 8.0
    DEFAULT_IMAGE_DURATION = 4.0
    TRANSITION_DURATION = 0.5
    ZOOM_INTENSITY = 1.3
    FACE_ZOOM_INTENSITY = 1.5

# 출력 형식 설정
if output_format == "vertical":
    Config.OUTPUT_WIDTH = 1080
    Config.OUTPUT_HEIGHT = 1920
    print("📐 출력 형식: 세로형 (1080x1920) - 릴스/틱톡/쇼츠용")
else:
    Config.OUTPUT_WIDTH = 1920
    Config.OUTPUT_HEIGHT = 1080
    print("📐 출력 형식: 가로형 (1920x1080) - YouTube/PC용")

# 폴더 생성
os.makedirs('uploads', exist_ok=True)
os.makedirs('output', exist_ok=True)

# ============================================================
# 4. 데이터 클래스 정의
# ============================================================
class CameraTechnique(Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    DOLLY_IN = "dolly_in"
    KEN_BURNS = "ken_burns"
    FACE_TRACK = "face_track"
    STATIC = "static"

class TransitionType(Enum):
    CUT = "cut"
    FADE = "fade"
    CROSSFADE = "crossfade"
    FLASH = "flash"

@dataclass
class MediaInfo:
    filepath: str
    media_type: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    faces: List[Dict] = field(default_factory=list)

@dataclass
class BeatInfo:
    beat_times: List[float]
    tempo: float
    strong_beats: List[float]
    total_duration: float

@dataclass
class EditDecision:
    media_index: int
    start_time: float
    end_time: float
    camera_technique: CameraTechnique
    transition_in: TransitionType
    zoom_target: Optional[Dict] = None

# ============================================================
# 5. 얼굴 감지기
# ============================================================
class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def detect(self, image):
        if image is None:
            return []
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)
            faces = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    faces.append({
                        'relative_center': (bbox.xmin + bbox.width/2, bbox.ymin + bbox.height/2),
                        'relative_size': (bbox.width, bbox.height),
                        'confidence': det.score[0]
                    })
            return sorted(faces, key=lambda x: x['confidence'], reverse=True)
        except:
            return []

face_detector = FaceDetector()

# ============================================================
# 6. 음악 분석
# ============================================================
def analyze_music(audio_path):
    print(f"\n🎵 음악 분석 중: {os.path.basename(audio_path)}")
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=np.mean(onset_env)*1.5, distance=sr//10)
    strong_beats = librosa.frames_to_time(peaks, sr=sr).tolist()

    tempo_val = float(tempo[0]) if hasattr(tempo, '__iter__') else float(tempo)

    print(f"   ✓ 길이: {duration:.1f}초")
    print(f"   ✓ 템포: {tempo_val:.1f} BPM")
    print(f"   ✓ 비트: {len(beat_times)}개, 강한 비트: {len(strong_beats)}개")

    return BeatInfo(beat_times, tempo_val, strong_beats, duration)

# ============================================================
# 7. 미디어 분석
# ============================================================
def analyze_media(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    # 이미지
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif']:
        img = cv2.imread(filepath)
        if img is None:
            print(f"   ⚠️ 이미지 읽기 실패: {os.path.basename(filepath)}")
            return None
        h, w = img.shape[:2]
        faces = face_detector.detect(img)
        print(f"   🖼️ {os.path.basename(filepath)} ({w}x{h}, 얼굴 {len(faces)}개)")
        return MediaInfo(filepath, 'image', Config.DEFAULT_IMAGE_DURATION, w, h, faces)

    # 비디오
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"   ⚠️ 비디오 읽기 실패: {os.path.basename(filepath)}")
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        ret, frame = cap.read()
        faces = face_detector.detect(frame) if ret else []
        cap.release()
        print(f"   🎬 {os.path.basename(filepath)} ({w}x{h}, {duration:.1f}초, 얼굴 {len(faces)}개)")
        return MediaInfo(filepath, 'video', duration, w, h, faces)

    # 오디오
    elif ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']:
        print(f"   🎵 {os.path.basename(filepath)}")
        return MediaInfo(filepath, 'audio')

    return None

# ============================================================
# 8. 이징 함수
# ============================================================
def ease_in_out(t):
    return t * t * (3 - 2 * t)

# ============================================================
# 9. 카메라 효과
# ============================================================
def apply_camera_effect(clip, technique, face_info=None):
    duration = clip.duration
    if duration <= 0:
        return clip

    target_x = face_info['relative_center'][0] if face_info else 0.5
    target_y = face_info['relative_center'][1] if face_info else 0.5

    if technique == CameraTechnique.STATIC:
        return clip

    # 줌인 / 얼굴 추적
    if technique in [CameraTechnique.ZOOM_IN, CameraTechnique.FACE_TRACK]:
        zoom_max = Config.FACE_ZOOM_INTENSITY if technique == CameraTechnique.FACE_TRACK else Config.ZOOM_INTENSITY
        def effect(get_frame, t):
            progress = ease_in_out(t / duration)
            scale = 1.0 + (zoom_max - 1.0) * progress
            frame = get_frame(t)
            fh, fw = frame.shape[:2]
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            cx, cy = int(target_x * nw), int(target_y * nh)
            x1 = max(0, min(nw - fw, cx - fw // 2))
            y1 = max(0, min(nh - fh, cy - fh // 2))
            return resized[y1:y1+fh, x1:x1+fw]
        return clip.fl(effect)

    # 줌아웃
    if technique == CameraTechnique.ZOOM_OUT:
        def effect(get_frame, t):
            progress = ease_in_out(t / duration)
            scale = Config.ZOOM_INTENSITY - (Config.ZOOM_INTENSITY - 1.0) * progress
            frame = get_frame(t)
            fh, fw = frame.shape[:2]
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            cx, cy = int(target_x * nw), int(target_y * nh)
            x1 = max(0, min(nw - fw, cx - fw // 2))
            y1 = max(0, min(nh - fh, cy - fh // 2))
            return resized[y1:y1+fh, x1:x1+fw]
        return clip.fl(effect)

    # 패닝
    if technique in [CameraTechnique.PAN_LEFT, CameraTechnique.PAN_RIGHT]:
        scale = 1.2
        def effect(get_frame, t):
            progress = ease_in_out(t / duration)
            frame = get_frame(t)
            fh, fw = frame.shape[:2]
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            max_off = nw - fw
            x = int(max_off * (1-progress)) if technique == CameraTechnique.PAN_LEFT else int(max_off * progress)
            y = (nh - fh) // 2
            return resized[y:y+fh, x:x+fw]
        return clip.fl(effect)

    # 켄 번스
    if technique == CameraTechnique.KEN_BURNS:
        s1 = random.choice([1.0, 1.25])
        s2 = 1.25 if s1 == 1.0 else 1.0
        def effect(get_frame, t):
            progress = ease_in_out(t / duration)
            scale = s1 + (s2 - s1) * progress
            frame = get_frame(t)
            fh, fw = frame.shape[:2]
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            cx, cy = int(target_x * nw), int(target_y * nh)
            x1 = max(0, min(nw - fw, cx - fw // 2))
            y1 = max(0, min(nh - fh, cy - fh // 2))
            return resized[y1:y1+fh, x1:x1+fw]
        return clip.fl(effect)

    # 돌리 인
    if technique == CameraTechnique.DOLLY_IN:
        def effect(get_frame, t):
            progress = ease_in_out(t / duration)
            scale = 1.0 + 0.15 * progress
            frame = get_frame(t)
            fh, fw = frame.shape[:2]
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            x, y = (nw - fw) // 2, (nh - fh) // 2
            return resized[y:y+fh, x:x+fw]
        return clip.fl(effect)

    return clip

# ============================================================
# 10. 크기 맞추기 (검정 여백)
# ============================================================
def fit_to_output(clip):
    tw, th = Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT
    cw, ch = clip.size

    target_ratio = tw / th
    clip_ratio = cw / ch

    if abs(clip_ratio - target_ratio) < 0.01:
        return clip.resize((tw, th))

    # 가로가 더 넓음 -> 위아래 검정
    if clip_ratio > target_ratio:
        new_w = tw
        new_h = int(tw / clip_ratio)
        resized = clip.resize((new_w, new_h))
        y_off = (th - new_h) // 2
        bg = ColorClip((tw, th), color=(0,0,0)).set_duration(clip.duration)
        return CompositeVideoClip([bg, resized.set_position(("center", y_off))]).set_duration(clip.duration)
    # 세로가 더 넓음 -> 좌우 검정
    else:
        new_h = th
        new_w = int(th * clip_ratio)
        resized = clip.resize((new_w, new_h))
        x_off = (tw - new_w) // 2
        bg = ColorClip((tw, th), color=(0,0,0)).set_duration(clip.duration)
        return CompositeVideoClip([bg, resized.set_position((x_off, "center"))]).set_duration(clip.duration)

# ============================================================
# 11. 편집 계획 생성
# ============================================================
def create_edit_plan(media_list, beat_info):
    decisions = []
    current_time = 0.0
    idx = 0

    print("\n🎬 편집 계획 생성 중...")

    while current_time < beat_info.total_duration and idx < len(media_list) * 3:
        media = media_list[idx % len(media_list)]

        # 클립 길이
        clip_dur = None
        for bt in beat_info.strong_beats:
            if bt > current_time + Config.MIN_CLIP_DURATION:
                clip_dur = bt - current_time
                break
        if not clip_dur:
            clip_dur = (60.0 / beat_info.tempo) * random.choice([2, 3, 4])

        clip_dur = max(Config.MIN_CLIP_DURATION, min(Config.MAX_CLIP_DURATION, clip_dur))
        if media.media_type == 'video':
            clip_dur = min(clip_dur, media.duration)
        if current_time + clip_dur > beat_info.total_duration:
            clip_dur = beat_info.total_duration - current_time
        if clip_dur < 0.5:
            break

        has_face = len(media.faces) > 0
        is_strong = any(abs(bt - current_time) < 0.3 for bt in beat_info.strong_beats)

        # 카메라 기법
        if has_face:
            tech = CameraTechnique.FACE_TRACK if is_strong else random.choice([
                CameraTechnique.ZOOM_IN, CameraTechnique.ZOOM_OUT, CameraTechnique.KEN_BURNS
            ])
        else:
            tech = random.choice([
                CameraTechnique.KEN_BURNS, CameraTechnique.PAN_LEFT,
                CameraTechnique.PAN_RIGHT, CameraTechnique.DOLLY_IN
            ])

        # 트랜지션
        if len(decisions) == 0:
            trans = TransitionType.FADE
        elif is_strong:
            trans = random.choice([TransitionType.CUT, TransitionType.FLASH])
        else:
            trans = random.choice([TransitionType.CUT, TransitionType.CROSSFADE])

        zoom_target = media.faces[0] if media.faces else None

        decisions.append(EditDecision(
            idx % len(media_list), current_time, current_time + clip_dur,
            tech, trans, zoom_target
        ))

        print(f"   [{len(decisions):2d}] {current_time:5.1f}s ~ {current_time+clip_dur:5.1f}s | {tech.value:12s} | {os.path.basename(media.filepath)}")

        current_time += clip_dur - Config.TRANSITION_DURATION
        idx += 1

    return decisions

# ============================================================
# 12. 비디오 렌더링
# ============================================================
def render_video(media_list, decisions, audio_path, output_path):
    print(f"\n🎬 비디오 렌더링 시작...")
    print(f"   출력: {Config.OUTPUT_WIDTH}x{Config.OUTPUT_HEIGHT}")

    clips = []
    total = len(decisions)

    for i, dec in enumerate(decisions):
        print(f"   클립 처리 중: {i+1}/{total}", end='\r')
        media = media_list[dec.media_index]
        dur = dec.end_time - dec.start_time

        try:
            # 클립 생성
            if media.media_type == 'image':
                clip = ImageClip(media.filepath).set_duration(dur)
            else:
                clip = VideoFileClip(media.filepath)
                if clip.duration > dur:
                    st = (clip.duration - dur) / 2
                    clip = clip.subclip(st, st + dur)
                elif clip.duration < dur:
                    clip = clip.loop(duration=dur)

            # 크기 맞추기
            clip = fit_to_output(clip)

            # 카메라 효과
            clip = apply_camera_effect(clip, dec.camera_technique, dec.zoom_target)

            # 트랜지션
            if dec.transition_in == TransitionType.FADE:
                clip = clip.fadein(0.5)
            elif dec.transition_in == TransitionType.FLASH:
                clip = clip.fadein(0.2)
            elif dec.transition_in == TransitionType.CROSSFADE and clips:
                clip = clip.crossfadein(0.5)

            clips.append(clip)

        except Exception as e:
            print(f"\n   ⚠️ 클립 {i+1} 오류: {e}")

    if not clips:
        print("\n❌ 처리된 클립이 없습니다!")
        return None

    print(f"\n   ✓ {len(clips)}개 클립 처리 완료")
    print("   클립 연결 중...")

    final = clips[0] if len(clips) == 1 else concatenate_videoclips(clips, method="compose")

    print("   오디오 추가 중...")
    audio = AudioFileClip(audio_path)
    final = final.set_audio(audio.subclip(0, min(audio.duration, final.duration)))

    print(f"   파일 저장 중... (시간이 좀 걸립니다)")
    final.write_videofile(
        output_path,
        fps=Config.OUTPUT_FPS,
        codec='libx264',
        audio_codec='aac',
        preset='medium',
        threads=4,
        verbose=False,
        logger=None
    )

    # 정리
    final.close()
    audio.close()
    for c in clips:
        try:
            c.close()
        except:
            pass

    print(f"\n✅ 렌더링 완료!")
    return output_path

# ============================================================
# 13. 메인 실행
# ============================================================
print("\n" + "="*60)
print("🎬 AI 자동 비디오 편집기 v2.0")
print("="*60)

# 파일 업로드
print("\n📁 파일을 업로드하세요!")
print("   - 이미지/비디오 + 음악 파일을 함께 선택하세요")
print("   - 여러 파일을 한번에 선택할 수 있습니다 (Ctrl+클릭)")
print("   - 음악 파일(mp3, wav 등)은 필수입니다!\n")

uploaded = files.upload()

if not uploaded:
    print("❌ 업로드된 파일이 없습니다!")
else:
    # 파일 저장 및 분석
    print("\n" + "="*60)
    print("📊 미디어 분석")
    print("="*60)

    media_list = []
    audio_path = None
    beat_info = None

    for filename, content in uploaded.items():
        filepath = os.path.join('uploads', filename)
        with open(filepath, 'wb') as f:
            f.write(content)

        info = analyze_media(filepath)
        if info:
            if info.media_type == 'audio':
                audio_path = filepath
                beat_info = analyze_music(filepath)
            else:
                media_list.append(info)

    # 검증
    if not audio_path:
        print("\n❌ 오류: 음악 파일이 없습니다! mp3, wav 등을 업로드하세요.")
    elif not media_list:
        print("\n❌ 오류: 이미지나 비디오 파일이 없습니다!")
    else:
        # 분석 결과
        print("\n" + "="*60)
        print("📋 분석 결과")
        print("="*60)
        img_cnt = sum(1 for m in media_list if m.media_type == 'image')
        vid_cnt = sum(1 for m in media_list if m.media_type == 'video')
        print(f"   🖼️ 이미지: {img_cnt}개")
        print(f"   🎬 비디오: {vid_cnt}개")
        print(f"   🎵 음악: {beat_info.total_duration:.1f}초 ({beat_info.tempo:.0f} BPM)")
        print(f"   📐 출력: {Config.OUTPUT_WIDTH}x{Config.OUTPUT_HEIGHT}")

        # 편집 계획
        decisions = create_edit_plan(media_list, beat_info)

        # 렌더링
        output_path = 'output/auto_edited_video.mp4'
        result = render_video(media_list, decisions, audio_path, output_path)

        if result:
            # 다운로드
            print("\n" + "="*60)
            print("📥 다운로드")
            print("="*60)
            files.download(output_path)

            # 미리보기
            print("\n🎬 미리보기:")
            try:
                from base64 import b64encode
                mp4_data = open(output_path, 'rb').read()
                data_url = "data:video/mp4;base64," + b64encode(mp4_data).decode()
                display(HTML(f'''
                <video width="640" controls autoplay>
                    <source src="{data_url}" type="video/mp4">
                </video>
                '''))
            except Exception as e:
                print(f"   미리보기 실패: {e}")
                print("   다운로드된 파일을 확인하세요!")

print("\n" + "="*60)
print("🎬 완료!")
print("="*60)
