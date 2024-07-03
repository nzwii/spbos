import rawpy
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

ref_path = Path("data/240703_ref.DNG")
jet_path = Path("data/240703_jet.MOV")
output_gif_path = Path("output/bos_output.gif")
output_mp4_path = Path("output/bos_output.mp4")

# RAW画像の処理？
def postproccesing(path):
    raw = rawpy.imread(str(path))
    return raw.postprocess(
        gamma=[1.0, 1.0],
        no_auto_bright=True,
        output_color=rawpy.ColorSpace.raw,
        use_camera_wb=True,
        use_auto_wb=False,
        output_bps=16,
        no_auto_scale=True
    )

# ref画像を読み込んでグレースケールに変換
ref_img = postproccesing(ref_path)
# ref_gray = ref_img[:, :, 1]

# jet動画をフレームごとに読み込み
cap = cv2.VideoCapture(str(jet_path))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
jet_gray_example = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
jet_height, jet_width = jet_gray_example.shape
# Resize ref_img to match the resolution of jet frames
ref_gray = cv2.resize(ref_img[:, :, 1], (jet_width, jet_height))

# 各フレームについてref画像との差分を計算
# LPF（低域通過フィルタ）を適用
frames = []
# Reset video to the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Prepare for MP4 writing
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (jet_width, jet_height), False)

for _ in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    jet_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bos = (jet_gray - ref_gray) * np.gradient(ref_gray, axis=0)
    
    lpf = cv2.blur(bos, (100, 100))
    lpf -= np.min(lpf)
    lpf /= np.amax(lpf)
    lpf_img = (lpf * 255).astype(np.uint8)
    
    frames.append(Image.fromarray(lpf_img))
    out.write(lpf_img)

cap.release()
out.release()

# Save the GIF
frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], loop=0, duration=int(1000 / fps))

print(f"✅ GIF saved to {output_gif_path}")
print(f"✅ MP4 saved to {output_mp4_path}")