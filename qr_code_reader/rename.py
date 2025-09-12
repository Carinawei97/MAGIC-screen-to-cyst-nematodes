import cv2
import os
import time
import random
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
from concurrent.futures import ProcessPoolExecutor, TimeoutError

folder_path = '/path/to/files'
dirsToSkip = ['empty']

def sanitize_filename(s):
    s = str(s).strip().replace('\n','_').replace('\r','_')
    s = s.replace(' ', '_').replace(',', '-').replace(';', '-').replace('"', '')
    return s

def try_quick_detect_multi(img):
    detector = cv2.QRCodeDetector()
    scales = [1, 1.5, 2]
    rotations = [i * 9 for i in range(40)]

    for scale in scales:
        resized = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"  Trying scale {scale}")
        for angle in rotations:
            rotated = resized if angle == 0 else cv2.warpAffine(
                resized,
                cv2.getRotationMatrix2D((resized.shape[1]//2, resized.shape[0]//2), angle, 1),
                (resized.shape[1], resized.shape[0])
            )
            print(f"    Trying rotation {angle} degrees")
            data, _, _ = detector.detectAndDecode(rotated)
            if data:
                print(f"      Detected OpenCV: {data}")
                return [data]  # break immediately
            for d in pyzbar_decode(rotated, symbols=[ZBarSymbol.QRCODE]):
                s = d.data.decode('utf-8')
                if s:
                    print(f"      Detected pyzbar: {s}")
                    return [s]  # break immediately
    return []

def try_all_ways(img, max_seconds=100):
    start = time.time()
    detector = cv2.QRCodeDetector()

    # 1. Whole image
    codes = try_quick_detect_multi(img)
    if codes:
        print(f"  QR found in whole image: {codes[0]}")
        return codes[0]

    # 2. Bounding box
    _, pts = detector.detect(img)
    if pts is not None:
        pts = pts[0].astype(int)
        x, y, w, h = cv2.boundingRect(pts)
        crop = img[y:y+h, x:x+w]
        codes = try_quick_detect_multi(crop)
        if codes:
            print(f"  QR found in cropped bounding box: {codes[0]}")
            return codes[0]

    # 3. Sliding window
    window_size = 1024
    step = 256
    h, w = img.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            elapsed = time.time() - start
            if elapsed > max_seconds:
                print(f"  Timeout reached after {elapsed:.1f}s")
                return None

            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)
            win = img[y:y_end, x:x_end]

            codes = try_quick_detect_multi(win)
            if codes:
                print(f"  QR found in sliding window at x={x}, y={y}: {codes[0]}")
                return codes[0]

    print(f"  No QR code found after full search ({time.time()-start:.1f}s)")
    return None

def _unique_dst(dst):
    if not os.path.exists(dst):
        return dst
    base, ext = os.path.splitext(dst)
    i = 1
    while True:
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1

def process_file(path):
    path = os.path.abspath(path)
    fn = os.path.basename(path)
    dir_path = os.path.dirname(path)

    if not fn.lower().endswith(('.png', '.bmp')):
        return
    if fn.startswith('failed_'):
        return

    img = cv2.imread(path)
    if img is None:
        new_name = f"failed_{fn}"
    else:
        qr = try_all_ways(img)
        if qr:
            qr_clean = sanitize_filename(qr)
            new_name = f"{qr_clean}_{fn}"
        else:
            new_name = f"failed_{fn}"

    dst = os.path.join(dir_path, new_name)
    dst = _unique_dst(dst)

    print(f"Attempting rename:\nFrom: {path}\nTo:   {dst}\n")
    try:
        os.rename(path, dst)
        print(f"***** RENAMED *****\nFrom: {path}\nTo:   {dst}\n******************\n")
    except Exception as e:
        print(f"FAILED to rename {path}: {e}")

def main():
    imgs = []
    for R, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in dirsToSkip and not d.startswith('.')]
        for f in files:
            if f.lower().endswith(('.png', '.bmp')):
                imgs.append(os.path.abspath(os.path.join(R, f)))

    random.shuffle(imgs)
    print(f"Found {len(imgs)} images to process.")

    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(process_file, p) for p in imgs]
        for f in futures:
            try:
                f.result(timeout=100)
            except TimeoutError:
                print("TIMEOUT")

if __name__ == "__main__":
    main()

