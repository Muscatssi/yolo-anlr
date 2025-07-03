from ultralytics import YOLO
from pytesseract import image_to_string
import numpy as np
import cv2
import os
import re

def show(img, title="Image", save_dir="yellow_result"):
    os.makedirs(save_dir, exist_ok=True)
    filename = title.replace(" ", "_") + ".jpg"
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, img)
    print(f"📸 저장됨: {path}")

def split_plate_by_ratio(img, upper_ratio=0.4, lower_start_ratio=0.3):
    h, w = img.shape[:2]
    upper_y = int(h * upper_ratio)
    lower_y = int(h * lower_start_ratio)

    upper = img[:upper_y, :]
    lower = img[lower_y:, :]  # ← 여기를 더 위부터 시작하게

    return upper, lower

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)

    return binary

def extract_plate_text(warped_img):
    # 1. 상/하단 분할
    upper, lower = split_plate_by_ratio(warped_img)

    # 2. 각각 전처리
    binary_upper = preprocess_for_ocr(upper)
    binary_lower = preprocess_for_ocr(lower)

    # 3. 시각화 (선택)
    show(binary_upper, "07. Upper (only number without region)")
    show(binary_lower, "08. Lower (char+number)")

    # 4. OCR 설정
    config_upper = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    config_lower = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789가나다라마바사아자하허호거너더러머버서어저고노도로모보소오조구누두루무부수우주배'

    # 5. OCR 실행
    upper_text = image_to_string(binary_upper, lang='kor', config=config_upper)
    lower_text = image_to_string(binary_lower, lang='kor', config=config_lower)

    # 6. 후처리 (한글 + 숫자만 남기기)
    upper_clean = re.sub(r'[^0-9]', '', upper_text)
    lower_clean = re.sub(r'[^가-힣0-9]', '', lower_text)

    # 7. 결과 출력
    print(f"🔹 상단 인식 결과: {upper_clean}")
    print(f"🔸 하단 인식 결과: {lower_clean}")
    #return upper_clean, lower_clean


image_path = "A01.JPG"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: 이미지를 찾을 수 없습니다! 경로: {image_path}")
else:
    print("1. 이미지 로드 성공.")

os.makedirs("threshold_cropped", exist_ok=True)

model = YOLO("license-plate-finetune-v1n.onnx")
results = model(img)

print("번호판을 탐지하는 중입니다...")
results = model(img)

# Process results list
for each_result in results:
    each_result.save(filename="threshold_cropped/result.jpg")  # save to disk

for i, result in enumerate(results):
    boxes = result.boxes  # Boxes 객체
    xyxy = boxes.xyxy.cpu().numpy().astype(int)  # (x1, y1, x2, y2) 좌표

    print(f"총 {len(xyxy)}개의 번호판을 찾았습니다. OCR을 시작합니다.")

    for j, box in enumerate(xyxy):
        # 1: 탐지 영역 자르기
        x1, y1, x2, y2 = box
        pixel = 4
        cropped_img = img[y1-pixel:y2+pixel, x1-pixel:x2+pixel]  # 이미지 자르기
        crop_path = f"threshold_cropped/{j}_original.jpg"

        show(cropped_img, "01. Original Image")

        # 2: 노란 번호판 강조
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        show(mask, "02. Yellow Mask")

        masked = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
        show(masked, "03. Yellow Region")
        
        # 4. 전처리
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        show(edges, "04. Canny Edges")

        # 5. 윤곽선 탐색 및 네 꼭짓점 추출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate = None

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and area > 500:
                candidate = approx
                break

        if candidate is not None:
            def order_points(pts):
                pts = pts.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)

                rect[0] = pts[np.argmin(s)]  # top-left
                rect[2] = pts[np.argmax(s)]  # bottom-right
                rect[1] = pts[np.argmin(diff)]  # top-right
                rect[3] = pts[np.argmax(diff)]  # bottom-left

                return rect
            corners = order_points(candidate)
            temp = cropped_img.copy()
            for pt in corners:
                cv2.circle(temp, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
            show(temp, "05. Detected Corner Points")

            # 6. 번호판 영역 잘라내기
            (tl, tr, br, bl) = corners
            width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(corners, dst_pts)
            warped = cv2.warpPerspective(cropped_img, M, (width, height))
            show(warped, "06. Warped Plate")

            # 7. OCR 문자 추출
            extract_plate_text(warped)

        else:
            print("번호판 후보를 찾지 못했습니다!")

print("\n모든 작업이 완료되었습니다.")
