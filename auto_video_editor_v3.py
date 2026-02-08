#@title 🎬 AI 자동 비디오 편집기 v3.0
#@markdown ---
#@markdown ### ⚙️ 출력 형식 선택
output_format = "horizontal"  #@param ["horizontal", "vertical"]
#@markdown - **horizontal**: 가로형 (1920x1080) - YouTube, PC
#@markdown - **vertical**: 세로형 (1080x1920) - 릴스, 틱톡, 쇼츠
#@markdown ---
#@markdown ### 실행하면 업로드 버튼이 나타납니다
#@markdown 1. **[파일 추가]** 버튼으로 파일 업로드 (여러 번 가능)
#@markdown 2. 모든 파일 업로드 후 **[생성 시작]** 버튼 클릭
#@markdown ---

# ============================================================
# 1. 설치
# ============================================================
print("="*60)
print("📦 라이브러리 설치 중... (1~2분)")
print("="*60)

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'moviepy==1.0.3'], capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'librosa'], capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python-headless'], capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mediapipe'], capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scipy'], capture_output=True)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], capture_output=True)
print("✅ 설치 완료!\n")

# ============================================================
# 2. 임포트
# ============================================================
print("📚 로딩 중...")
import os, cv2, numpy as np, random, warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
warnings.filterwarnings('ignore')

import librosa
from scipy.signal import find_peaks
import mediapipe as mp
from google.colab import files
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, ColorClip,
    CompositeVideoClip, concatenate_videoclips
)
print("✅ 로드 완료!\n")

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

if output_format == "vertical":
    Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT = 1080, 1920
    print("📐 출력: 세로형 (1080x1920)")
else:
    Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT = 1920, 1080
    print("📐 출력: 가로형 (1920x1080)")

os.makedirs('uploads', exist_ok=True)
os.makedirs('output', exist_ok=True)

# ============================================================
# 4. 클래스 정의
# ============================================================
class CameraTechnique(Enum):
    ZOOM_IN="zoom_in"; ZOOM_OUT="zoom_out"; PAN_LEFT="pan_left"; PAN_RIGHT="pan_right"
    DOLLY_IN="dolly_in"; KEN_BURNS="ken_burns"; FACE_TRACK="face_track"; STATIC="static"

class TransitionType(Enum):
    CUT="cut"; FADE="fade"; CROSSFADE="crossfade"; FLASH="flash"

@dataclass
class MediaInfo:
    filepath: str; media_type: str; duration: float = 0.0
    width: int = 0; height: int = 0; faces: List[Dict] = field(default_factory=list)

@dataclass
class BeatInfo:
    beat_times: List[float]; tempo: float; strong_beats: List[float]; total_duration: float

@dataclass
class EditDecision:
    media_index: int; start_time: float; end_time: float
    camera_technique: CameraTechnique; transition_in: TransitionType
    zoom_target: Optional[Dict] = None

# ============================================================
# 5. 얼굴 감지
# ============================================================
class FaceDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    def detect(self, image):
        if image is None: return []
        try:
            results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.detections: return []
            faces = []
            for det in results.detections:
                b = det.location_data.relative_bounding_box
                faces.append({'relative_center':(b.xmin+b.width/2, b.ymin+b.height/2),
                              'relative_size':(b.width,b.height), 'confidence':det.score[0]})
            return sorted(faces, key=lambda x:x['confidence'], reverse=True)
        except: return []

face_detector = FaceDetector()

# ============================================================
# 6. 분석 함수들
# ============================================================
def analyze_music(path):
    y, sr = librosa.load(path)
    dur = librosa.get_duration(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset, height=np.mean(onset)*1.5, distance=sr//10)
    strong = librosa.frames_to_time(peaks, sr=sr).tolist()
    t = float(tempo[0]) if hasattr(tempo,'__iter__') else float(tempo)
    return BeatInfo(beat_times, t, strong, dur)

def analyze_media(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg','.jpeg','.png','.bmp','.webp','.gif']:
        img = cv2.imread(path)
        if img is None: return None
        h,w = img.shape[:2]
        faces = face_detector.detect(img)
        return MediaInfo(path,'image',Config.DEFAULT_IMAGE_DURATION,w,h,faces)
    elif ext in ['.mp4','.avi','.mov','.mkv','.webm','.m4v']:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return None
        w,h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(5); dur = int(cap.get(7))/fps if fps>0 else 0
        ret,frame = cap.read()
        faces = face_detector.detect(frame) if ret else []
        cap.release()
        return MediaInfo(path,'video',dur,w,h,faces)
    elif ext in ['.mp3','.wav','.ogg','.m4a','.flac','.aac']:
        return MediaInfo(path,'audio')
    return None

# ============================================================
# 7. 효과 함수들
# ============================================================
def ease(t): return t*t*(3-2*t)

def apply_effect(clip, tech, face=None):
    dur = clip.duration
    if dur <= 0: return clip
    tx = face['relative_center'][0] if face else 0.5
    ty = face['relative_center'][1] if face else 0.5

    if tech == CameraTechnique.STATIC: return clip

    if tech in [CameraTechnique.ZOOM_IN, CameraTechnique.FACE_TRACK]:
        zm = Config.FACE_ZOOM_INTENSITY if tech==CameraTechnique.FACE_TRACK else Config.ZOOM_INTENSITY
        def fx(gf,t):
            s=1+(zm-1)*ease(t/dur); f=gf(t); fh,fw=f.shape[:2]
            nw,nh=int(fw*s),int(fh*s); r=cv2.resize(f,(nw,nh))
            cx,cy=int(tx*nw),int(ty*nh)
            x1,y1=max(0,min(nw-fw,cx-fw//2)),max(0,min(nh-fh,cy-fh//2))
            return r[y1:y1+fh,x1:x1+fw]
        return clip.fl(fx)

    if tech == CameraTechnique.ZOOM_OUT:
        def fx(gf,t):
            s=Config.ZOOM_INTENSITY-(Config.ZOOM_INTENSITY-1)*ease(t/dur); f=gf(t); fh,fw=f.shape[:2]
            nw,nh=int(fw*s),int(fh*s); r=cv2.resize(f,(nw,nh))
            cx,cy=int(tx*nw),int(ty*nh)
            x1,y1=max(0,min(nw-fw,cx-fw//2)),max(0,min(nh-fh,cy-fh//2))
            return r[y1:y1+fh,x1:x1+fw]
        return clip.fl(fx)

    if tech in [CameraTechnique.PAN_LEFT, CameraTechnique.PAN_RIGHT]:
        def fx(gf,t):
            p=ease(t/dur); f=gf(t); fh,fw=f.shape[:2]
            nw,nh=int(fw*1.2),int(fh*1.2); r=cv2.resize(f,(nw,nh))
            mo=nw-fw; x=int(mo*(1-p)) if tech==CameraTechnique.PAN_LEFT else int(mo*p)
            y=(nh-fh)//2; return r[y:y+fh,x:x+fw]
        return clip.fl(fx)

    if tech == CameraTechnique.KEN_BURNS:
        s1,s2 = (1.0,1.25) if random.random()>0.5 else (1.25,1.0)
        def fx(gf,t):
            s=s1+(s2-s1)*ease(t/dur); f=gf(t); fh,fw=f.shape[:2]
            nw,nh=int(fw*s),int(fh*s); r=cv2.resize(f,(nw,nh))
            cx,cy=int(tx*nw),int(ty*nh)
            x1,y1=max(0,min(nw-fw,cx-fw//2)),max(0,min(nh-fh,cy-fh//2))
            return r[y1:y1+fh,x1:x1+fw]
        return clip.fl(fx)

    if tech == CameraTechnique.DOLLY_IN:
        def fx(gf,t):
            s=1+0.15*ease(t/dur); f=gf(t); fh,fw=f.shape[:2]
            nw,nh=int(fw*s),int(fh*s); r=cv2.resize(f,(nw,nh))
            x,y=(nw-fw)//2,(nh-fh)//2; return r[y:y+fh,x:x+fw]
        return clip.fl(fx)

    return clip

def fit_output(clip):
    tw,th = Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT
    cw,ch = clip.size
    tr,cr = tw/th, cw/ch
    if abs(cr-tr)<0.01: return clip.resize((tw,th))
    if cr > tr:
        nw,nh = tw, int(tw/cr)
        r = clip.resize((nw,nh)); yo=(th-nh)//2
        bg = ColorClip((tw,th),color=(0,0,0)).set_duration(clip.duration)
        return CompositeVideoClip([bg,r.set_position(("center",yo))]).set_duration(clip.duration)
    else:
        nw,nh = int(th*cr), th
        r = clip.resize((nw,nh)); xo=(tw-nw)//2
        bg = ColorClip((tw,th),color=(0,0,0)).set_duration(clip.duration)
        return CompositeVideoClip([bg,r.set_position((xo,"center"))]).set_duration(clip.duration)

# ============================================================
# 8. 편집 & 렌더링
# ============================================================
def create_plan(media, beat):
    decs=[]; ct=0.0; idx=0
    while ct < beat.total_duration and idx < len(media)*3:
        m = media[idx % len(media)]
        cd = None
        for bt in beat.strong_beats:
            if bt > ct + Config.MIN_CLIP_DURATION: cd = bt - ct; break
        if not cd: cd = (60/beat.tempo)*random.choice([2,3,4])
        cd = max(Config.MIN_CLIP_DURATION, min(Config.MAX_CLIP_DURATION, cd))
        if m.media_type=='video': cd = min(cd, m.duration)
        if ct + cd > beat.total_duration: cd = beat.total_duration - ct
        if cd < 0.5: break
        hf = len(m.faces)>0
        st = any(abs(bt-ct)<0.3 for bt in beat.strong_beats)
        tech = CameraTechnique.FACE_TRACK if hf and st else random.choice([CameraTechnique.ZOOM_IN,CameraTechnique.ZOOM_OUT,CameraTechnique.KEN_BURNS]) if hf else random.choice([CameraTechnique.KEN_BURNS,CameraTechnique.PAN_LEFT,CameraTechnique.PAN_RIGHT,CameraTechnique.DOLLY_IN])
        tr = TransitionType.FADE if not decs else (random.choice([TransitionType.CUT,TransitionType.FLASH]) if st else random.choice([TransitionType.CUT,TransitionType.CROSSFADE]))
        zt = m.faces[0] if m.faces else None
        decs.append(EditDecision(idx%len(media),ct,ct+cd,tech,tr,zt))
        ct += cd - Config.TRANSITION_DURATION; idx += 1
    return decs

def render(media, decs, audio_path, out_path, progress_label):
    clips = []
    for i,d in enumerate(decs):
        progress_label.value = f"🎬 렌더링: {i+1}/{len(decs)} 클립 처리 중..."
        m = media[d.media_index]; dur = d.end_time - d.start_time
        try:
            if m.media_type=='image': c = ImageClip(m.filepath).set_duration(dur)
            else:
                c = VideoFileClip(m.filepath)
                if c.duration > dur: st=(c.duration-dur)/2; c=c.subclip(st,st+dur)
                elif c.duration < dur: c=c.loop(duration=dur)
            c = fit_output(c)
            c = apply_effect(c, d.camera_technique, d.zoom_target)
            if d.transition_in==TransitionType.FADE: c=c.fadein(0.5)
            elif d.transition_in==TransitionType.FLASH: c=c.fadein(0.2)
            elif d.transition_in==TransitionType.CROSSFADE and clips: c=c.crossfadein(0.5)
            clips.append(c)
        except: pass

    if not clips: return None

    progress_label.value = "🎬 클립 연결 중..."
    final = clips[0] if len(clips)==1 else concatenate_videoclips(clips, method="compose")

    progress_label.value = "🎬 오디오 추가 중..."
    audio = AudioFileClip(audio_path)
    final = final.set_audio(audio.subclip(0, min(audio.duration, final.duration)))

    progress_label.value = "🎬 파일 저장 중... (시간이 걸립니다)"
    final.write_videofile(out_path, fps=Config.OUTPUT_FPS, codec='libx264', audio_codec='aac',
                          preset='medium', threads=4, verbose=False, logger=None)
    final.close(); audio.close()
    for c in clips:
        try: c.close()
        except: pass
    return out_path

# ============================================================
# 9. 업로드 매니저 UI
# ============================================================
class UploadManager:
    def __init__(self):
        self.file_paths = []
        self.file_info = []

        # UI 위젯 생성
        self.status = widgets.HTML(value="<b>📁 업로드된 파일: 0개</b>")
        self.file_list = widgets.HTML(value="")
        self.progress = widgets.HTML(value="")

        self.add_btn = widgets.Button(description="📁 파일 추가", button_style='info',
                                       layout=widgets.Layout(width='150px', height='40px'))
        self.start_btn = widgets.Button(description="🎬 생성 시작", button_style='success',
                                         layout=widgets.Layout(width='150px', height='40px'))
        self.start_btn.disabled = True

        self.add_btn.on_click(self.on_add_click)
        self.start_btn.on_click(self.on_start_click)

        self.output_area = widgets.Output()

    def on_add_click(self, b):
        with self.output_area:
            clear_output(wait=True)
            print("📤 파일 선택 창이 열립니다...")
            print("   여러 파일 선택: Ctrl+클릭")
            try:
                uploaded = files.upload()
                for fn, content in uploaded.items():
                    fp = os.path.join('uploads', fn)
                    with open(fp, 'wb') as f:
                        f.write(content)
                    self.file_paths.append(fp)

                    # 파일 타입 확인
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in ['.mp3','.wav','.ogg','.m4a','.flac','.aac']:
                        ftype = "🎵"
                    elif ext in ['.mp4','.avi','.mov','.mkv','.webm','.m4v']:
                        ftype = "🎬"
                    else:
                        ftype = "🖼️"
                    self.file_info.append(f"{ftype} {fn}")

                self.update_display()
                print(f"\n✅ {len(uploaded)}개 파일 추가됨!")
            except Exception as e:
                print(f"업로드 취소 또는 오류: {e}")

    def update_display(self):
        # 상태 업데이트
        cnt = len(self.file_paths)
        self.status.value = f"<b>📁 업로드된 파일: {cnt}개</b>"

        # 파일 목록 업데이트
        if self.file_info:
            list_html = "<div style='background:#f0f0f0; padding:10px; border-radius:5px; margin:5px 0;'>"
            for info in self.file_info:
                list_html += f"{info}<br>"
            list_html += "</div>"
            self.file_list.value = list_html

        # 시작 버튼 활성화 체크
        has_audio = any(os.path.splitext(f)[1].lower() in ['.mp3','.wav','.ogg','.m4a','.flac','.aac']
                       for f in self.file_paths)
        has_visual = any(os.path.splitext(f)[1].lower() in ['.jpg','.jpeg','.png','.bmp','.webp','.gif',
                                                            '.mp4','.avi','.mov','.mkv','.webm','.m4v']
                        for f in self.file_paths)
        self.start_btn.disabled = not (has_audio and has_visual)

        if not has_audio and cnt > 0:
            self.progress.value = "<span style='color:orange'>⚠️ 음악 파일(mp3 등)을 추가하세요!</span>"
        elif not has_visual and cnt > 0:
            self.progress.value = "<span style='color:orange'>⚠️ 이미지/비디오 파일을 추가하세요!</span>"
        else:
            self.progress.value = ""

    def on_start_click(self, b):
        self.add_btn.disabled = True
        self.start_btn.disabled = True
        self.progress.value = "<b>🎬 영상 생성 시작...</b>"

        with self.output_area:
            clear_output(wait=True)
            self.generate_video()

    def generate_video(self):
        print("="*60)
        print("📊 미디어 분석 중...")
        print("="*60)

        media_list = []
        audio_path = None
        beat_info = None

        for fp in self.file_paths:
            info = analyze_media(fp)
            if info:
                if info.media_type == 'audio':
                    audio_path = fp
                    print(f"🎵 음악: {os.path.basename(fp)}")
                    beat_info = analyze_music(fp)
                    print(f"   → {beat_info.total_duration:.1f}초, {beat_info.tempo:.0f} BPM")
                elif info.media_type == 'image':
                    print(f"🖼️ 이미지: {os.path.basename(fp)} ({info.width}x{info.height}, 얼굴 {len(info.faces)}개)")
                    media_list.append(info)
                else:
                    print(f"🎬 비디오: {os.path.basename(fp)} ({info.width}x{info.height}, {info.duration:.1f}초)")
                    media_list.append(info)

        print(f"\n📋 총 {len(media_list)}개 미디어 + 음악 {beat_info.total_duration:.1f}초")

        print("\n" + "="*60)
        print("🎬 편집 계획 생성...")
        print("="*60)

        decs = create_plan(media_list, beat_info)
        for i, d in enumerate(decs):
            m = media_list[d.media_index]
            print(f"[{i+1:2d}] {d.start_time:5.1f}s ~ {d.end_time:5.1f}s | {d.camera_technique.value:12s} | {os.path.basename(m.filepath)}")

        print("\n" + "="*60)
        print("🎬 렌더링...")
        print("="*60)

        out_path = 'output/auto_edited_video.mp4'
        result = render(media_list, decs, audio_path, out_path, self.progress)

        if result:
            self.progress.value = "<b style='color:green'>✅ 완료!</b>"
            print("\n✅ 렌더링 완료!")

            print("\n📥 다운로드:")
            files.download(out_path)

            print("\n🎬 미리보기:")
            try:
                from base64 import b64encode
                data = b64encode(open(out_path,'rb').read()).decode()
                display(HTML(f'<video width="640" controls><source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))
            except:
                print("다운로드된 파일을 확인하세요!")
        else:
            self.progress.value = "<b style='color:red'>❌ 실패</b>"
            print("❌ 렌더링 실패!")

        self.add_btn.disabled = False

    def show(self):
        title = widgets.HTML(value="""
        <h2>🎬 AI 자동 비디오 편집기 v3.0</h2>
        <p>1. <b>[파일 추가]</b> 버튼으로 파일 업로드 (여러 번 가능)</p>
        <p>2. 이미지/비디오 + 음악 파일이 모두 있으면 <b>[생성 시작]</b> 활성화</p>
        <p>3. <b>[생성 시작]</b> 클릭하면 자동 편집!</p>
        <hr>
        """)

        buttons = widgets.HBox([self.add_btn, self.start_btn])

        ui = widgets.VBox([
            title,
            self.status,
            self.file_list,
            buttons,
            self.progress,
            self.output_area
        ])

        display(ui)

# ============================================================
# 10. 실행
# ============================================================
manager = UploadManager()
manager.show()
