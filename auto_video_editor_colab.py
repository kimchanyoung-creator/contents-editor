#@title 🎬 AI 자동 비디오 편집기 - Google Colab용
#@markdown ### 사용법: 이 전체 코드를 Colab 셀에 붙여넣고 실행하세요!

# ============================================================
# 1. 라이브러리 설치
# ============================================================
print("📦 라이브러리 설치 중...")
import subprocess
subprocess.run(['pip', 'install', '-q', 'moviepy==1.0.3', 'librosa', 'opencv-python-headless', 'mediapipe', 'pillow', 'scipy'])
subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'])
print("✅ 설치 완료!")

# ============================================================
# 2. 라이브러리 임포트
# ============================================================
import os
import cv2
import numpy as np
import librosa
import mediapipe as mp
from PIL import Image
from google.colab import files
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, vfx
)
from scipy.signal import find_peaks
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

print("✅ 라이브러리 로드 완료!")

# ============================================================
# 3. 설정값
# ============================================================
class Config:
    OUTPUT_WIDTH = 1920
    OUTPUT_HEIGHT = 1080
    OUTPUT_FPS = 30
    MIN_CLIP_DURATION = 1.5
    MAX_CLIP_DURATION = 8.0
    DEFAULT_IMAGE_DURATION = 4.0
    TRANSITION_DURATION = 0.5
    ZOOM_INTENSITY = 1.3
    FACE_ZOOM_INTENSITY = 1.5

# ============================================================
# 4. 카메라 기법 & 트랜지션 정의
# ============================================================
class CameraTechnique(Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
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
    has_motion: bool = False

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
# 5. 얼굴 감지
# ============================================================
class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                cx = bbox.xmin + bbox.width / 2
                cy = bbox.ymin + bbox.height / 2
                faces.append({
                    'relative_center': (cx, cy),
                    'relative_size': (bbox.width, bbox.height),
                    'confidence': det.score[0]
                })
        return sorted(faces, key=lambda x: x['confidence'], reverse=True)

face_detector = FaceDetector()

# ============================================================
# 6. 음악 분석
# ============================================================
def analyze_music(audio_path):
    print(f"🎵 음악 분석 중: {os.path.basename(audio_path)}")
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=np.mean(onset_env) * 1.5, distance=sr//10)
    strong_beats = librosa.frames_to_time(peaks, sr=sr).tolist()

    tempo_val = float(tempo[0]) if hasattr(tempo, '__iter__') else float(tempo)

    print(f"  템포: {tempo_val:.1f} BPM | 비트: {len(beat_times)}개 | 강한비트: {len(strong_beats)}개")
    return BeatInfo(beat_times, tempo_val, strong_beats, duration)

# ============================================================
# 7. 미디어 분석
# ============================================================
def analyze_media(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        print(f"🖼️ 이미지 분석: {os.path.basename(filepath)}")
        img = cv2.imread(filepath)
        h, w = img.shape[:2]
        faces = face_detector.detect(img)
        print(f"  크기: {w}x{h} | 얼굴: {len(faces)}개")
        return MediaInfo(filepath, 'image', Config.DEFAULT_IMAGE_DURATION, w, h, faces)

    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        print(f"🎬 비디오 분석: {os.path.basename(filepath)}")
        cap = cv2.VideoCapture(filepath)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        ret, frame = cap.read()
        faces = face_detector.detect(frame) if ret else []
        cap.release()
        print(f"  크기: {w}x{h} | 길이: {duration:.1f}초 | 얼굴: {len(faces)}개")
        return MediaInfo(filepath, 'video', duration, w, h, faces)

    elif ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
        return MediaInfo(filepath, 'audio')

    return None

# ============================================================
# 8. 카메라 효과 적용
# ============================================================
def ease_in_out(t):
    return t * t * (3 - 2 * t)

def apply_camera_effect(clip, technique, face_info=None):
    duration = clip.duration
    w, h = clip.size

    target_x = face_info['relative_center'][0] if face_info else 0.5
    target_y = face_info['relative_center'][1] if face_info else 0.5

    if technique == CameraTechnique.STATIC:
        return clip

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

    if technique in [CameraTechnique.PAN_LEFT, CameraTechnique.PAN_RIGHT]:
        scale = 1.2
        def effect(get_frame, t):
            progress = ease_in_out(t / duration)
            frame = get_frame(t)
            fh, fw = frame.shape[:2]
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            max_offset = nw - fw
            x = int(max_offset * (1 - progress)) if technique == CameraTechnique.PAN_LEFT else int(max_offset * progress)
            y = (nh - fh) // 2
            return resized[y:y+fh, x:x+fw]
        return clip.fl(effect)

    if technique == CameraTechnique.KEN_BURNS:
        start_scale = random.choice([1.0, 1.25])
        end_scale = 1.25 if start_scale == 1.0 else 1.0
        def effect(get_frame, t):
            progress = ease_in_out(t / duration)
            scale = start_scale + (end_scale - start_scale) * progress
            frame = get_frame(t)
            fh, fw = frame.shape[:2]
            nw, nh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))
            cx, cy = int(target_x * nw), int(target_y * nh)
            x1 = max(0, min(nw - fw, cx - fw // 2))
            y1 = max(0, min(nh - fh, cy - fh // 2))
            return resized[y1:y1+fh, x1:x1+fw]
        return clip.fl(effect)

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
# 9. 클립 크기 맞추기
# ============================================================
def fit_to_output(clip):
    tw, th = Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT
    cw, ch = clip.size
    target_ratio = tw / th
    clip_ratio = cw / ch

    if clip_ratio > target_ratio:
        clip = clip.resize(height=th)
        nw = int(clip_ratio * th)
        x = (nw - tw) // 2
        clip = clip.crop(x1=x, x2=x+tw, y1=0, y2=th)
    else:
        clip = clip.resize(width=tw)
        nh = int(tw / clip_ratio)
        y = (nh - th) // 2
        clip = clip.crop(x1=0, x2=tw, y1=y, y2=y+th)
    return clip

# ============================================================
# 10. 자동 편집 결정
# ============================================================
def create_edit_decisions(media_list, beat_info):
    decisions = []
    current_time = 0.0
    media_idx = 0

    print("\n🎬 편집 계획 생성 중...")

    while current_time < beat_info.total_duration and media_idx < len(media_list) * 3:
        media = media_list[media_idx % len(media_list)]

        # 클립 길이 결정
        clip_dur = None
        for bt in beat_info.strong_beats:
            if bt > current_time + Config.MIN_CLIP_DURATION:
                clip_dur = bt - current_time
                break
        if not clip_dur:
            beat_interval = 60.0 / beat_info.tempo
            clip_dur = beat_interval * random.choice([2, 3, 4])

        clip_dur = max(Config.MIN_CLIP_DURATION, min(Config.MAX_CLIP_DURATION, clip_dur))
        if media.media_type == 'video':
            clip_dur = min(clip_dur, media.duration)

        if current_time + clip_dur > beat_info.total_duration:
            clip_dur = beat_info.total_duration - current_time
        if clip_dur < 0.5:
            break

        # 카메라 기법 선택
        has_face = len(media.faces) > 0
        is_strong = any(abs(bt - current_time) < 0.3 for bt in beat_info.strong_beats)

        if has_face:
            if is_strong:
                technique = CameraTechnique.FACE_TRACK
            else:
                technique = random.choice([CameraTechnique.ZOOM_IN, CameraTechnique.ZOOM_OUT, CameraTechnique.KEN_BURNS])
        else:
            technique = random.choice([CameraTechnique.KEN_BURNS, CameraTechnique.PAN_LEFT, CameraTechnique.PAN_RIGHT, CameraTechnique.DOLLY_IN])

        # 트랜지션 선택
        if len(decisions) == 0:
            trans = TransitionType.FADE
        elif is_strong:
            trans = random.choice([TransitionType.CUT, TransitionType.FLASH])
        else:
            trans = random.choice([TransitionType.CUT, TransitionType.CROSSFADE])

        zoom_target = media.faces[0] if media.faces else None

        decisions.append(EditDecision(
            media_idx % len(media_list),
            current_time,
            current_time + clip_dur,
            technique,
            trans,
            zoom_target
        ))

        print(f"  [{len(decisions)}] {current_time:.1f}s-{current_time+clip_dur:.1f}s: {technique.value} | {os.path.basename(media.filepath)}")

        current_time += clip_dur - Config.TRANSITION_DURATION
        media_idx += 1

    return decisions

# ============================================================
# 11. 비디오 렌더링
# ============================================================
def render_video(media_list, decisions, audio_path, output_path="output_video.mp4"):
    print("\n🎬 비디오 렌더링 시작...")

    clips = []
    for i, dec in enumerate(decisions):
        print(f"  클립 {i+1}/{len(decisions)} 처리 중...")
        media = media_list[dec.media_index]
        clip_dur = dec.end_time - dec.start_time

        # 클립 생성
        if media.media_type == 'image':
            clip = ImageClip(media.filepath).set_duration(clip_dur)
        else:
            clip = VideoFileClip(media.filepath)
            if clip.duration > clip_dur:
                start = (clip.duration - clip_dur) / 2
                clip = clip.subclip(start, start + clip_dur)
            elif clip.duration < clip_dur:
                clip = clip.loop(duration=clip_dur)

        # 크기 맞추기
        clip = fit_to_output(clip)

        # 카메라 효과
        clip = apply_camera_effect(clip, dec.camera_technique, dec.zoom_target)

        # 트랜지션
        if dec.transition_in == TransitionType.FADE:
            clip = clip.fadein(0.5)
        elif dec.transition_in == TransitionType.CROSSFADE and clips:
            clip = clip.crossfadein(0.5)
        elif dec.transition_in == TransitionType.FLASH:
            clip = clip.fadein(0.2)

        clips.append(clip)

    # 연결
    print("  클립 연결 중...")
    if len(clips) == 1:
        final = clips[0]
    else:
        final = concatenate_videoclips(clips, method="compose")

    # 오디오 추가
    print("  오디오 추가 중...")
    audio = AudioFileClip(audio_path)
    final = final.set_audio(audio.subclip(0, min(audio.duration, final.duration)))

    # 저장
    print(f"  저장 중: {output_path}")
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

    final.close()
    audio.close()
    for c in clips:
        c.close()

    print(f"\n✅ 완료: {output_path}")
    return output_path

# ============================================================
# 12. 메인 실행
# ============================================================
def main():
    print("\n" + "🎬"*25)
    print("      AI 자동 비디오 편집기")
    print("🎬"*25 + "\n")

    # 폴더 생성
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # 파일 업로드
    print("📁 파일을 업로드하세요!")
    print("   - 이미지: jpg, png, webp 등")
    print("   - 비디오: mp4, mov, avi 등")
    print("   - 음악: mp3, wav, m4a 등 (필수!)\n")

    uploaded = files.upload()

    # 파일 저장 및 분석
    media_list = []
    audio_path = None
    beat_info = None

    print("\n" + "="*50)
    print("📊 미디어 분석")
    print("="*50)

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

    if not audio_path:
        print("❌ 오류: 음악 파일이 필요합니다!")
        return
    if not media_list:
        print("❌ 오류: 이미지 또는 비디오 파일이 필요합니다!")
        return

    print("\n" + "="*50)
    print("📋 분석 결과")
    print("="*50)
    print(f"  미디어: {len(media_list)}개")
    print(f"  음악 길이: {beat_info.total_duration:.1f}초")
    print(f"  템포: {beat_info.tempo:.1f} BPM")

    # 편집 결정 생성
    decisions = create_edit_decisions(media_list, beat_info)

    # 렌더링
    output_path = os.path.join('output', 'auto_edited_video.mp4')
    render_video(media_list, decisions, audio_path, output_path)

    # 다운로드
    print("\n" + "="*50)
    print("📥 비디오 다운로드")
    print("="*50)
    files.download(output_path)

    # 미리보기
    from IPython.display import HTML
    from base64 import b64encode
    mp4 = open(output_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f'<video width="800" controls><source src="{data_url}" type="video/mp4"></video>'))

# 실행!
main()
