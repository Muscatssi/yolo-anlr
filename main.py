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
        pixel = 5
        cropped_img = img[y1-pixel:y2+pixel, x1-pixel:x2+pixel]  # 이미지 자르기
        crop_path = f"cropped/crop_{j}_original.jpg"
        cv2.imwrite(crop_path, cropped_img)

        # 2: 4개의 꼭짓점 찾기
        resized = cv2.resize(cropped_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        crop_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        crop_blur = cv2.GaussianBlur(crop_gray, (0, 0), 5)
        #sharpened = cv2.addWeighted(crop_blur, 1.5, crop_blur, -0.5, 0)
        #crop_canny = cv2.Canny(crop_blur, 30, 90)
        _, crop_thred = cv2.threshold(crop_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        thin = cv2.erode(crop_thred, kernel, iterations=1)

        #cv2.imwrite(f"cropped/canny_{j}.jpg", crop_canny)
        cv2.imwrite(f"cropped/threshold_{j}.jpg", crop_thred)

        #contours, _ = cv2.findContours(crop_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(crop_thred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"   - 경고({j+1}): Canny Edge에서 윤곽선을 찾지 못했습니다.")
            continue
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        contour_vis_img = cropped_img.copy()
        cv2.drawContours(contour_vis_img, contours, -1, (0,255,0),2)
        cv2.imwrite(f"cropped/contours_{j}.jpg", contour_vis_img) 

        # 가장 큰 윤곽선을 번호판 후보로 선택
        largest_contour = contours[0]
        
        # 1. 최소 면적 사각형 찾기
        rect = cv2.minAreaRect(largest_contour)
        
        # 2. 사각형의 4개 꼭짓점 좌표 구하기
        box = cv2.boxPoints(rect)
        box = np.intp(box) # 정수형으로 변환

        # 찾은 사각형(파란색)을 시각화 이미지에 그려서 확인
        cv2.drawContours(contour_vis_img, [box], 0, (255, 0, 0), 2)
        cv2.imwrite(f"cropped/crop_{j}_contours_and_box.jpg", contour_vis_img)
        
        plate_contour = box # plate_contour 변수에 꼭짓점 좌표 할당

        """
        plate_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            if len(approx) == 4:
                plate_contour = approx
                break
        """

        # 3: 꼭짓점 기준 원근 변환
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
        print(f"   - 번호판({j}) 원근 변환 성공!")
        cv2.imwrite(f"cropped/warped_{j}.jpg", warped_plate)
        final_plate_for_ocr = warped_plate
        
        # 4: OCR 인식
        gray_ocr = cv2.cvtColor(final_plate_for_ocr, cv2.COLOR_BGR2GRAY)

        # OCR 전처리
        resized_ocr = cv2.resize(gray_ocr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, binary_plate = cv2.threshold(resized_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        """
        # 노이즈 제거
        kernel = np.ones((1,1), np.uint8)
        binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, kernel)
        binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, kernel)
"""
        cv2.imwrite(f"cropped/binary_for_ocr_{j}.jpg", binary_plate)
        

        # Pytesseract로 OCR 수행
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주바사아자배하허호'
        plate_text = image_to_string(binary_plate, lang='kor', config=custom_config)
        
        cleaned_text = re.sub(r'[^가-힣0-9]', '', plate_text)
        print(f"   - 👁️ 인식 결과({j+1}): {cleaned_text}")

print("\n모든 작업이 완료되었습니다.")