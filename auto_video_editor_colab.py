#@title 🎬 AI 자동 비디오 편집기 v2.0 - Google Colab용
#@markdown ### 사용법: 이 전체 코드를 Colab 셀에 붙여넣고 실행하세요!

# ============================================================
# 1. 라이브러리 설치
# ============================================================
print("📦 라이브러리 설치 중...")
import subprocess
subprocess.run(['pip', 'install', '-q', 'moviepy==1.0.3', 'librosa', 'opencv-python-headless', 'mediapipe', 'pillow', 'scipy'])
subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], capture_output=True)
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
    VideoFileClip, ImageClip, AudioFileClip, ColorClip,
    CompositeVideoClip, concatenate_videoclips, vfx
)
from scipy.signal import find_peaks
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import warnings
import time
warnings.filterwarnings('ignore')

print("✅ 라이브러리 로드 완료!")

# ============================================================
# 3. 설정값
# ============================================================
class Config:
    # 가로형 (16:9)
    HORIZONTAL_WIDTH = 1920
    HORIZONTAL_HEIGHT = 1080
    # 세로형 (9:16)
    VERTICAL_WIDTH = 1080
    VERTICAL_HEIGHT = 1920

    # 현재 설정 (기본값: 가로형)
    OUTPUT_WIDTH = 1920
    OUTPUT_HEIGHT = 1080
    OUTPUT_FPS = 30

    MIN_CLIP_DURATION = 1.5
    MAX_CLIP_DURATION = 8.0
    DEFAULT_IMAGE_DURATION = 4.0
    TRANSITION_DURATION = 0.5
    ZOOM_INTENSITY = 1.3
    FACE_ZOOM_INTENSITY = 1.5

    @classmethod
    def set_horizontal(cls):
        cls.OUTPUT_WIDTH = cls.HORIZONTAL_WIDTH
        cls.OUTPUT_HEIGHT = cls.HORIZONTAL_HEIGHT
        print(f"📐 출력 형식: 가로형 (1920x1080)")

    @classmethod
    def set_vertical(cls):
        cls.OUTPUT_WIDTH = cls.VERTICAL_WIDTH
        cls.OUTPUT_HEIGHT = cls.VERTICAL_HEIGHT
        print(f"📐 출력 형식: 세로형 (1080x1920)")

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
        if image is None:
            return []
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

    print(f"  ✓ 템포: {tempo_val:.1f} BPM | 비트: {len(beat_times)}개 | 강한비트: {len(strong_beats)}개")
    return BeatInfo(beat_times, tempo_val, strong_beats, duration)

# ============================================================
# 7. 미디어 분석
# ============================================================
def analyze_media(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif']:
        print(f"🖼️ 이미지: {os.path.basename(filepath)}", end=" ")
        img = cv2.imread(filepath)
        if img is None:
            print("(읽기 실패)")
            return None
        h, w = img.shape[:2]
        faces = face_detector.detect(img)
        print(f"({w}x{h}, 얼굴 {len(faces)}개)")
        return MediaInfo(filepath, 'image', Config.DEFAULT_IMAGE_DURATION, w, h, faces)

    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']:
        print(f"🎬 비디오: {os.path.basename(filepath)}", end=" ")
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print("(읽기 실패)")
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        ret, frame = cap.read()
        faces = face_detector.detect(frame) if ret else []
        cap.release()
        print(f"({w}x{h}, {duration:.1f}초, 얼굴 {len(faces)}개)")
        return MediaInfo(filepath, 'video', duration, w, h, faces)

    elif ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']:
        print(f"🎵 오디오: {os.path.basename(filepath)}")
        return MediaInfo(filepath, 'audio')

    return None

# ============================================================
# 8. 카메라 효과 적용
# ============================================================
def ease_in_out(t):
    return t * t * (3 - 2 * t)

def apply_camera_effect(clip, technique, face_info=None):
    duration = clip.duration
    if duration <= 0:
        return clip

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
# 9. 클립 크기 맞추기 (검정 여백 추가 - letterbox/pillarbox)
# ============================================================
def fit_to_output_with_padding(clip):
    """미디어 비율을 유지하면서 출력 크기에 맞추고, 빈 공간은 검정색으로 채움"""
    tw, th = Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT
    cw, ch = clip.size

    target_ratio = tw / th
    clip_ratio = cw / ch

    if abs(clip_ratio - target_ratio) < 0.01:
        # 비율이 거의 같으면 단순 리사이즈
        return clip.resize((tw, th))

    if clip_ratio > target_ratio:
        # 원본이 더 넓음 (위아래 검정 여백 - letterbox)
        new_width = tw
        new_height = int(tw / clip_ratio)
        resized_clip = clip.resize((new_width, new_height))
        y_offset = (th - new_height) // 2

        # 검정 배경 생성
        background = ColorClip(size=(tw, th), color=(0, 0, 0)).set_duration(clip.duration)
        final = CompositeVideoClip([
            background,
            resized_clip.set_position(("center", y_offset))
        ])
    else:
        # 원본이 더 좁음 (좌우 검정 여백 - pillarbox)
        new_height = th
        new_width = int(th * clip_ratio)
        resized_clip = clip.resize((new_width, new_height))
        x_offset = (tw - new_width) // 2

        # 검정 배경 생성
        background = ColorClip(size=(tw, th), color=(0, 0, 0)).set_duration(clip.duration)
        final = CompositeVideoClip([
            background,
            resized_clip.set_position((x_offset, "center"))
        ])

    return final.set_duration(clip.duration)

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

        has_face = len(media.faces) > 0
        is_strong = any(abs(bt - current_time) < 0.3 for bt in beat_info.strong_beats)

        if has_face:
            if is_strong:
                technique = CameraTechnique.FACE_TRACK
            else:
                technique = random.choice([CameraTechnique.ZOOM_IN, CameraTechnique.ZOOM_OUT, CameraTechnique.KEN_BURNS])
        else:
            technique = random.choice([CameraTechnique.KEN_BURNS, CameraTechnique.PAN_LEFT, CameraTechnique.PAN_RIGHT, CameraTechnique.DOLLY_IN])

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

        print(f"  [{len(decisions):2d}] {current_time:5.1f}s ~ {current_time+clip_dur:5.1f}s | {technique.value:12s} | {os.path.basename(media.filepath)}")

        current_time += clip_dur - Config.TRANSITION_DURATION
        media_idx += 1

    return decisions

# ============================================================
# 11. 비디오 렌더링
# ============================================================
def render_video(media_list, decisions, audio_path, output_path="output_video.mp4"):
    print("\n🎬 비디오 렌더링 시작...")
    print(f"   출력 크기: {Config.OUTPUT_WIDTH}x{Config.OUTPUT_HEIGHT}")

    clips = []
    for i, dec in enumerate(decisions):
        print(f"  클립 {i+1}/{len(decisions)} 처리 중...", end="\r")
        media = media_list[dec.media_index]
        clip_dur = dec.end_time - dec.start_time

        try:
            if media.media_type == 'image':
                clip = ImageClip(media.filepath).set_duration(clip_dur)
            else:
                clip = VideoFileClip(media.filepath)
                if clip.duration > clip_dur:
                    start = (clip.duration - clip_dur) / 2
                    clip = clip.subclip(start, start + clip_dur)
                elif clip.duration < clip_dur:
                    clip = clip.loop(duration=clip_dur)

            # 크기 맞추기 (검정 여백 추가)
            clip = fit_to_output_with_padding(clip)

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
        except Exception as e:
            print(f"\n  ⚠️ 클립 {i+1} 처리 중 오류: {e}")
            continue

    if not clips:
        print("❌ 처리된 클립이 없습니다!")
        return None

    print(f"\n  ✓ {len(clips)}개 클립 처리 완료")
    print("  클립 연결 중...")

    if len(clips) == 1:
        final = clips[0]
    else:
        final = concatenate_videoclips(clips, method="compose")

    print("  오디오 추가 중...")
    audio = AudioFileClip(audio_path)
    final = final.set_audio(audio.subclip(0, min(audio.duration, final.duration)))

    print(f"  파일 저장 중: {output_path}")
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
        try:
            c.close()
        except:
            pass

    print(f"\n✅ 렌더링 완료: {output_path}")
    return output_path

# ============================================================
# 12. 파일 업로드 매니저
# ============================================================
class UploadManager:
    def __init__(self):
        self.uploaded_files = []
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('output', exist_ok=True)

    def upload_files(self):
        """파일 업로드 (여러 번 반복 가능)"""
        print("\n" + "="*60)
        print("📁 파일 업로드")
        print("="*60)
        print("지원 형식:")
        print("  🖼️ 이미지: jpg, png, webp, gif 등")
        print("  🎬 비디오: mp4, mov, avi, mkv 등")
        print("  🎵 오디오: mp3, wav, m4a, flac 등 (필수!)")
        print()
        print("💡 파일을 여러 번에 나눠서 업로드할 수 있습니다.")
        print("   업로드가 끝나면 아래에서 '완료'를 선택하세요.")
        print("="*60)

        upload_count = 0

        while True:
            print(f"\n현재 업로드된 파일: {len(self.uploaded_files)}개")
            print("-" * 40)
            print("1. 파일 추가 업로드")
            print("2. 업로드 완료 → 영상 생성 시작")
            print("-" * 40)

            try:
                choice = input("선택 (1 또는 2): ").strip()
            except:
                choice = "1"

            if choice == "2":
                if len(self.uploaded_files) == 0:
                    print("⚠️ 업로드된 파일이 없습니다. 파일을 먼저 업로드하세요.")
                    continue

                # 오디오 파일 확인
                has_audio = any(
                    os.path.splitext(f)[1].lower() in ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']
                    for f in self.uploaded_files
                )
                if not has_audio:
                    print("⚠️ 음악 파일이 필요합니다! 오디오 파일을 업로드하세요.")
                    continue

                # 이미지 또는 비디오 확인
                has_visual = any(
                    os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
                    for f in self.uploaded_files
                )
                if not has_visual:
                    print("⚠️ 이미지 또는 비디오 파일이 필요합니다!")
                    continue

                print("\n✅ 업로드 완료!")
                break

            else:
                # 파일 업로드
                print("\n📤 파일 선택 창이 열립니다...")
                print("   (여러 파일을 한번에 선택할 수 있습니다)")

                try:
                    uploaded = files.upload()

                    for filename, content in uploaded.items():
                        filepath = os.path.join('uploads', filename)
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        self.uploaded_files.append(filepath)
                        print(f"  ✓ {filename}")
                        upload_count += 1

                    print(f"\n📊 이번에 {len(uploaded)}개 파일 추가됨 (총 {len(self.uploaded_files)}개)")

                except Exception as e:
                    print(f"업로드 중 오류: {e}")
                    print("다시 시도해주세요.")

        return self.uploaded_files

    def get_file_list(self):
        """업로드된 파일 목록 반환"""
        return self.uploaded_files

# ============================================================
# 13. 메인 실행
# ============================================================
def main():
    print("\n" + "🎬"*30)
    print("       AI 자동 비디오 편집기 v2.0")
    print("🎬"*30 + "\n")

    # 1. 출력 형식 선택
    print("="*60)
    print("📐 출력 형식 선택")
    print("="*60)
    print("1. 가로형 (1920x1080) - YouTube, PC용")
    print("2. 세로형 (1080x1920) - 인스타 릴스, 틱톡, 쇼츠용")
    print("="*60)

    while True:
        try:
            format_choice = input("선택 (1 또는 2): ").strip()
        except:
            format_choice = "1"

        if format_choice == "1":
            Config.set_horizontal()
            break
        elif format_choice == "2":
            Config.set_vertical()
            break
        else:
            print("1 또는 2를 입력해주세요.")

    # 2. 파일 업로드
    upload_manager = UploadManager()
    uploaded_files = upload_manager.upload_files()

    # 3. 미디어 분석
    print("\n" + "="*60)
    print("📊 미디어 분석")
    print("="*60)

    media_list = []
    audio_path = None
    beat_info = None

    for filepath in uploaded_files:
        info = analyze_media(filepath)
        if info:
            if info.media_type == 'audio':
                audio_path = filepath
                beat_info = analyze_music(filepath)
            else:
                media_list.append(info)

    # 4. 분석 결과 출력
    print("\n" + "="*60)
    print("📋 분석 결과")
    print("="*60)

    image_count = sum(1 for m in media_list if m.media_type == 'image')
    video_count = sum(1 for m in media_list if m.media_type == 'video')

    print(f"  🖼️ 이미지: {image_count}개")
    print(f"  🎬 비디오: {video_count}개")
    print(f"  🎵 음악 길이: {beat_info.total_duration:.1f}초")
    print(f"  🎼 템포: {beat_info.tempo:.1f} BPM")
    print(f"  📐 출력: {Config.OUTPUT_WIDTH}x{Config.OUTPUT_HEIGHT}")

    # 5. 편집 결정 생성
    decisions = create_edit_decisions(media_list, beat_info)

    # 6. 렌더링
    output_path = os.path.join('output', 'auto_edited_video.mp4')
    result = render_video(media_list, decisions, audio_path, output_path)

    if result is None:
        print("❌ 비디오 생성에 실패했습니다.")
        return

    # 7. 다운로드
    print("\n" + "="*60)
    print("📥 비디오 다운로드")
    print("="*60)
    files.download(output_path)

    # 8. 미리보기
    print("\n🎬 미리보기:")
    from IPython.display import HTML, display
    from base64 import b64encode

    try:
        mp4 = open(output_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        display(HTML(f'<video width="640" controls><source src="{data_url}" type="video/mp4"></video>'))
    except Exception as e:
        print(f"미리보기 로드 실패: {e}")
        print("다운로드된 파일을 확인해주세요.")

# 실행!
if __name__ == "__main__":
    main()
