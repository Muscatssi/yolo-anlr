from ultralytics import YOLO
from pytesseract import image_to_string
import numpy as np
import cv2
import os
import re

def order_points(pts):
    """
    네 개의 꼭짓점(사각형)을 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 순서로 정렬합니다.
    C++ 코드의 꼭짓점 정렬 로직과 동일한 방식입니다.
    - 왼쪽 위: x+y 합이 가장 작음
    - 오른쪽 아래: x+y 합이 가장 큼
    - 오른쪽 위: y-x 차가 가장 작음
    - 왼쪽 아래: y-x 차가 가장 큼
    """
    # 4x2 크기의 배열 초기화
    rect = np.zeros((4, 2), dtype="float32")

    # x + y 합 계산
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 왼쪽 위
    rect[2] = pts[np.argmax(s)] # 오른쪽 아래

    # y - x 차 계산
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 오른쪽 위
    rect[3] = pts[np.argmax(diff)] # 왼쪽 아래

    return rect

image_path = "./cars.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: 이미지를 찾을 수 없습니다! 경로: {image_path}")
else:
    print("1. 이미지 로드 성공.")

os.makedirs("cropped", exist_ok=True)

model = YOLO("license-plate-finetune-v1n.onnx")
results = model(img)

print("번호판을 탐지하는 중입니다...")
results = model(img)

# Process results list
for each_result in results:
    """
    boxes = each_result.boxes  # Boxes object for bounding box outputs
    masks = each_result.masks  # Masks object for segmentation masks outputs
    keypoints = each_result.keypoints  # Keypoints object for pose outputs
    probs = each_result.probs  # Probs object for classification outputs
    obb = each_result.obb  # Oriented boxes object for OBB outputs
    # each_result.show()  # display to screen
    """
    each_result.save(filename="result.jpg")  # save to disk

for i, result in enumerate(results):
    boxes = result.boxes  # Boxes 객체
    xyxy = boxes.xyxy.cpu().numpy().astype(int)  # (x1, y1, x2, y2) 좌표

    print(f"총 {len(xyxy)}개의 번호판을 찾았습니다. OCR을 시작합니다.")

    for j, box in enumerate(xyxy):
        # 1: 탐지 영역 자르기
        x1, y1, x2, y2 = box
        cropped_img = img[y1:y2, x1:x2]  # 이미지 자르기
        crop_path = f"cropped/crop_{j}_original.jpg"
        cv2.imwrite(crop_path, cropped_img)

        # 2: 4개의 꼭짓점 찾기
        crop_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        crop_blur = cv2.GaussianBlur(crop_gray, (0,0), 3)
        #sharpened = cv2.addWeighted(resized_plate, 1.5, crop_blur, -0.5, 0)
        crop_canny = cv2.Canny(crop_blur, 100, 200)

        contours, _ = cv2.findContours(crop_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        #_, binary_plate = cv2.threshold(resized_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        plate_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            if len(approx) == 4:
                plate_contour = approx
                break

        # 3: 꼭짓점 찾았다면 원근 변환, 못 찾았으면 원본이미지 사용
        if plate_contour is not None:
            points = plate_contour.reshape(4, 2)
            src_pts = order_points(points)

            # 실제 번호판 비율로 조정 
            PLATE_WIDTH = 220
            PLATE_HEIGHT = 45

            dst_pts = np.array([[0, 0], 
                                [PLATE_WIDTH - 1, 0], 
                                [PLATE_WIDTH - 1, PLATE_HEIGHT - 1], 
                                [0, PLATE_HEIGHT - 1]], 
                                dtype="float32")
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # cropped_img 이미지에 대해 원근 변환을 적용합니다.
            warped_plate = cv2.warpPerspective(cropped_img, matrix, (PLATE_WIDTH, PLATE_HEIGHT))
            print(f"   - 번호판({j+1}) 원근 변환 성공!")
            cv2.imwrite(f"cropped/crop_{j}_warped.jpg", warped_plate)
            final_plate_for_ocr = warped_plate
        else:
            print(f"   - 경고({j+1}): 사각형을 찾지 못해 YOLO 영역을 그대로 사용합니다.")
            final_plate_for_ocr = cropped_img # 원근 변환 실패 시 YOLO 크롭 이미지를 사용

        # 4: OCR 인식
        gray_ocr = cv2.cvtColor(final_plate_for_ocr, cv2.COLOR_BGR2GRAY)

        # OCR 전처리
        resized_ocr = cv2.resize(gray_ocr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, binary_plate = cv2.threshold(resized_ocr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        kernel = np.ones((1,1), np.uint8)
        binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, kernel)
        binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(f"cropped/crop_{j}_binary_for_ocr.jpg", binary_plate)
        

        # Pytesseract로 OCR 수행
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주바사아자배하허호'
        plate_text = image_to_string(binary_plate, lang='kor', config=custom_config)
        
        cleaned_text = re.sub(r'[^가-힣0-9]', '', plate_text)
        print(f"   - 👁️ 인식 결과({j+1}): {cleaned_text}")

print("\n모든 작업이 완료되었습니다.")