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

    # 검출된 태그 주위에 사각형 그리기
    if len(detections) > 0:
        for detection in detections:
            # 꼭지점 좌표 추출
            rect = detection['lb-rb-rt-lt'].astype(int)
            # 사각형 그리기
            cv2.polylines(image, [rect], True, (0, 255, 0), 2)
            # 태그 번호 표시
            if 'id' in detection:  # AprilTag가 감지된 경우에만 ID를 표시합니다.
                cv2.putText(image, f"ID: {detection['id']}", (rect[0][0], rect[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # AprilTag의 중심 좌표 계산
                center_x = (rect[0][0] + rect[2][0]) // 2
                center_y = (rect[0][1] + rect[2][1]) // 2
                # 중심 좌표 출력
                cv2.putText(image, f"Center: ({center_x}, {center_y})", (rect[0][0], rect[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # AprilTag의 방향 계산
                direction = calculate_direction(rect)
                cv2.putText(image, f"Direction: {direction}", (rect[0][0], rect[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def calculate_direction(rect):
    # 좌우 변의 길이 계산
    left_width = np.linalg.norm(rect[0] - rect[1])
    right_width = np.linalg.norm(rect[2] - rect[3])

    # 좌우 변의 길이의 차이로 방향을 추정
    if abs(left_width - right_width) < 2:  # 길이의 차이가 10 이하이면 "Stay"로 설정
        direction = "Stay"
    elif left_width > right_width:
        direction = "Go Right"
    elif left_width < right_width:
        direction = "Go Left"
    else:
        direction = "Undefined"

    return direction

# RealSense 파이프라인 설정
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
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

