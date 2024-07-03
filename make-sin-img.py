import numpy as np
import cv2

# ディスプレイ
# for iPad 第6世代
ppi = 264
x_pix = 1536
y_pix = 2048
pix_per_mm = ppi / 25.4     # たぶん25.4(mm: 1インチ)は固定値
x_mm = x_pix / pix_per_mm
y_mm = y_pix / pix_per_mm
lam = 0.7 # mm

def sin_img(direction="x",bit=8):
    if direction == "x":
        x = np.linspace(0.0,x_mm,x_pix)
        sin = np.sin(2.0*np.pi*x/lam)
        img = np.tile(sin,(y_pix,1)).T
        img = 0.5 * (img + 1.0) * (2**bit-1.0)
    elif direction == "y":
        y = np.linspace(0.0,y_mm,y_pix)
        sin = np.sin(2.0*np.pi*y/lam)
        img = np.tile(sin,(x_pix,1))
        img = 0.5 * (img + 1.0) * (2**bit-1.0)
        
    if bit == 8:
        img = img.astype(np.uint8)
    elif bit == 16:
        img = img.astype(np.uint16)
    else:
        print("bit数は8か16")
    return img

img = sin_img(direction="x", bit=8)
cv2.imwrite("sin.png",img)