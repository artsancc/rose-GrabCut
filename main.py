import cv2
import numpy as np

img = cv2.imread("rose.jpeg")
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 80, 80])
upper_red1 = np.array([10, 255, 255])
# HSV'de kırmızının iki değeri olduğu için red2 değeri de eklendi.
lower_red2 = np.array([120, 30, 20])
upper_red2 = np.array([180, 255, 255])

kirmizi1 = cv2.inRange(hsv, lower_red1, upper_red1)
kirmizi2 = cv2.inRange(hsv, lower_red2, upper_red2)
gul_mask  = cv2.bitwise_or(kirmizi1, kirmizi2)

# GrabCut
maske      = np.zeros((h, w), np.uint8)
bgd_model  = np.zeros((1, 65), np.float64)
fgd_model  = np.zeros((1, 65), np.float64)
rect       = (70, 100, w - 140, h - 200)

cv2.grabCut(img, maske, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)

# Maskeleme
maske[gul_mask > 0] = cv2.GC_FGD
kenar = 45
maske[:kenar, :] = maske[-kenar:, :] = cv2.GC_BGD
maske[:, :kenar] = maske[:, -kenar:] = cv2.GC_BGD
maske[:int(h * 0.25), int(w * 0.25):int(w * 0.75)] = cv2.GC_BGD

cv2.grabCut(img, maske, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)

on_plan = np.where((maske == cv2.GC_FGD) | (maske == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

# Yeşil alan temizlemesi
yesil      = cv2.inRange(hsv, (35, 20, 10), (100, 255, 200))
dusuk_doy  = cv2.inRange(hsv, (0,   0,  0), (180,  55, 255))

on_plan[yesil > 0]     = 0
on_plan[dusuk_doy > 0] = 0
on_plan[gul_mask > 0]  = 255

# Maske gürültü temizleme.
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
on_plan = cv2.morphologyEx(on_plan, cv2.MORPH_OPEN,  kernel, iterations=2)
on_plan = cv2.morphologyEx(on_plan, cv2.MORPH_CLOSE, kernel, iterations=4)

_, labels, stats, _ = cv2.connectedComponentsWithStats(on_plan, connectivity=8)
en_buyuk = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
on_plan  = np.where(labels == en_buyuk, 255, 0).astype(np.uint8)
on_plan  = cv2.morphologyEx(on_plan, cv2.MORPH_CLOSE, kernel, iterations=3)

yumusak = cv2.GaussianBlur(on_plan, (9, 9), 0)

# Gülü mora çevir.
h_k, s_k, v_k = cv2.split(hsv)
alfa = yumusak.astype(np.float32) / 255.0

h_yeni = (np.full_like(h_k, 140) * alfa + h_k * (1 - alfa)).astype(np.uint8)  # 140 = mor
s_yeni = (np.clip(s_k.astype(np.int32) + 30, 0, 255) * alfa + s_k * (1 - alfa)).astype(np.uint8)

mor_gul = cv2.cvtColor(cv2.merge([h_yeni, s_yeni, v_k]), cv2.COLOR_HSV2BGR)

a3     = np.stack([alfa] * 3, axis=2)
sonuc  = (mor_gul.astype(np.float32) * a3).astype(np.uint8)

cv2.imwrite("mor_gul_siyah_bg.jpg", sonuc)

print("Tamamlandı!")