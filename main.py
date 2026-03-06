import cv2
import numpy as np


def goruntu_yukle(yol: str):
    """Görüntüyü yükler ve HSV dönüşümünü yapar."""
    img = cv2.imread(yol)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {yol}")
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv, h, w


def kirmizi_maske_olustur(hsv: np.ndarray) -> np.ndarray:
    """HSV görüntüsünden kırmızı renk maskesi üretir."""
    lower_red1 = np.array([0,   80,  80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([120,  30,  20])
    upper_red2 = np.array([180, 255, 255])

    kirmizi1 = cv2.inRange(hsv, lower_red1, upper_red1)
    kirmizi2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(kirmizi1, kirmizi2)


def grabcut_uygula(img: np.ndarray, gul_mask: np.ndarray,
                   h: int, w: int) -> np.ndarray:
    """GrabCut algoritması ile ön plan maskesi çıkarır."""
    maske     = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect      = (70, 100, w - 140, h - 200)

    # İlk GrabCut – dikdörtgen tabanlı başlatma
    cv2.grabCut(img, maske, rect, bgd_model, fgd_model, 8,
                cv2.GC_INIT_WITH_RECT)

    maske = maske_ipuclari_ekle(maske, gul_mask, h, w)

    cv2.grabCut(img, maske, rect, bgd_model, fgd_model, 10,
                cv2.GC_INIT_WITH_MASK)

    on_plan = np.where(
        (maske == cv2.GC_FGD) | (maske == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    return on_plan


def maske_ipuclari_ekle(maske: np.ndarray, gul_mask: np.ndarray,
                          h: int, w: int, kenar: int = 45) -> np.ndarray:
    """GrabCut maskesine ön plan / arka plan ipuçları ekler."""
    maske[gul_mask > 0] = cv2.GC_FGD

    # Kenar bölgelerini arka plan olarak işaretler.
    maske[:kenar, :]  = cv2.GC_BGD
    maske[-kenar:, :] = cv2.GC_BGD
    maske[:, :kenar]  = cv2.GC_BGD
    maske[:, -kenar:] = cv2.GC_BGD

    # Üst ve orta bölgeyi arka plan olarak işaretler.
    maske[:int(h * 0.25), int(w * 0.25):int(w * 0.75)] = cv2.GC_BGD

    return maske


def on_plan_temizle(on_plan: np.ndarray, hsv: np.ndarray,
                    gul_mask: np.ndarray) -> np.ndarray:
    """Yeşil ve düşük doygunluktaki pikselleri maskeden kaldırır."""
    yesil     = cv2.inRange(hsv, (35, 20,  10), (100, 255, 200))
    dusuk_doy = cv2.inRange(hsv, ( 0,  0,   0), (180,  55, 255))

    on_plan[yesil > 0]     = 0
    on_plan[dusuk_doy > 0] = 0
    on_plan[gul_mask > 0]  = 255   # Kesin kırmızı pikselleri geri ekle

    return on_plan


def morfoloji_uygula(on_plan: np.ndarray,
                     kernel_boyut: int = 15) -> np.ndarray:
    """Açma / kapama işlemleriyle maskedeki gürültüyü giderir."""
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_boyut, kernel_boyut)
    )
    on_plan = cv2.morphologyEx(on_plan, cv2.MORPH_OPEN,  kernel, iterations=2)
    on_plan = cv2.morphologyEx(on_plan, cv2.MORPH_CLOSE, kernel, iterations=4)
    return on_plan, kernel


def en_buyuk_bileseni_sec(on_plan: np.ndarray,
                           kernel: np.ndarray) -> np.ndarray:
    """Bağlantılı bileşenler arasından en büyük alanı seçer."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        on_plan, connectivity=8
    )
    en_buyuk = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    on_plan  = np.where(labels == en_buyuk, 255, 0).astype(np.uint8)
    on_plan  = cv2.morphologyEx(on_plan, cv2.MORPH_CLOSE, kernel, iterations=3)
    return on_plan


def gule_mor_renk_ver(hsv: np.ndarray, on_plan: np.ndarray) -> np.ndarray:
    """Maskelenen gül bölgesini mora çevirir, arka plan siyah kalır."""
    yumusak = cv2.GaussianBlur(on_plan, (9, 9), 0)
    alfa    = yumusak.astype(np.float32) / 255.0

    h_k, s_k, v_k = cv2.split(hsv)

    # Hue: 140 mor.
    h_yeni = (np.full_like(h_k, 140) * alfa + h_k * (1 - alfa)).astype(np.uint8)
    s_yeni = (
        np.clip(s_k.astype(np.int32) + 30, 0, 255) * alfa + s_k * (1 - alfa)
    ).astype(np.uint8)

    mor_gul = cv2.cvtColor(
        cv2.merge([h_yeni, s_yeni, v_k]), cv2.COLOR_HSV2BGR
    )

    # Arka planı siyah yapar.
    a3    = np.stack([alfa] * 3, axis=2)
    sonuc = (mor_gul.astype(np.float32) * a3).astype(np.uint8)

    return sonuc


def main(girdi_yolu: str = "rose.jpeg",
         cikti_yolu: str = "mor_gul_siyah_bg.jpg") -> None:
    """Ana işlem hattını sırayla çalıştırır."""
    img, hsv, h, w = goruntu_yukle(girdi_yolu)

    gul_mask = kirmizi_maske_olustur(hsv)
    on_plan  = grabcut_uygula(img, gul_mask, h, w)
    on_plan  = on_plan_temizle(on_plan, hsv, gul_mask)
    on_plan, kernel = morfoloji_uygula(on_plan)
    on_plan  = en_buyuk_bileseni_sec(on_plan, kernel)

    sonuc = gule_mor_renk_ver(hsv, on_plan)
    cv2.imwrite(cikti_yolu, sonuc)
    print(f"Tamamlandı → {cikti_yolu}")


if __name__ == "__main__":
    main()
