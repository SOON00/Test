import pyrealsense2 as rs
import numpy as np
import cv2
from apriltag import apriltag

def detect_apriltags(image):
    # AprilTag 검출기 생성
    detector = apriltag("tag36h11")

    # 회색조로 이미지 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # AprilTag 검출
    detections = detector.detect(gray)

    centers = []  # 중심 좌표를 저장할 리스트

    # 검출된 태그 주위에 사각형 그리기 (ID가 0인 경우만)
    if len(detections) > 0:
        for detection in detections:
            # ID가 0인 경우만 처리
            if 'id' in detection and detection['id'] == 0:
                # 꼭지점 좌표 추출
                rect = detection['lb-rb-rt-lt'].astype(int)
                # 사각형 그리기
                cv2.polylines(image, [rect], True, (0, 255, 0), 2)
                # 태그 번호 표시
                cv2.putText(image, f"ID: {detection['id']}", (rect[0][0], rect[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # AprilTag의 중심 좌표 계산
                center_x = (rect[0][0] + rect[2][0]) // 2
                center_y = (rect[0][1] + rect[2][1]) // 2
                centers.append((center_x, center_y))  # 중심 좌표 저장
                # 중심 좌표 출력
                cv2.putText(image, f"Center: ({center_x}, {center_y})", (rect[0][0], rect[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 중심 좌표가 4개인 경우 사각형 그리기 및 방향 계산
    if len(centers) == 4:
        centers = np.array(centers, dtype=np.int32)  # numpy 배열로 변환
        centers = order_points(centers)
        cv2.polylines(image, [centers], True, (0, 0, 255), 2)  # 빨간색 사각형 그리기
        
        # orientation 계산
        orientation = calculate_orientation(centers)
        cv2.putText(image, f"Orientation: {orientation:.2f}", (centers[0][0], centers[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="int")

    # 좌표의 합을 기준으로 좌상단과 우하단 찾기
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 좌표의 차이를 기준으로 우상단과 좌하단 찾기
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def calculate_orientation(centers):
    # 중심 좌표에서 첫 번째와 두 번째 포인트 사이의 벡터를 계산
    vector = centers[1] - centers[0]
    
    # 벡터의 각도를 계산
    angle = np.arctan2(vector[1], vector[0])
    
    # 각도를 도 단위로 변환
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

# RealSense 파이프라인 설정
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)

while True:
    # 프레임 캡처
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # 깊이 이미지와 컬러 이미지 가져오기
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # AprilTag 감지 및 방향 계산
    color_image_with_tags = detect_apriltags(color_image)

    # 이미지 표시
    cv2.imshow('RGB with AprilTag', color_image_with_tags)

    if cv2.waitKey(1) == ord('q'):
        break

# 파이프라인 정지 및 창 닫기
pipe.stop()
cv2.destroyAllWindows()

