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
                # AprilTag의 중심에 법선 벡터 표시
                cv2.arrowedLine(image, (center_x, center_y), (center_x, center_y - 50), (255, 0, 0), 2)

    return image

pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

pipe.start(cfg)

while True:
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                     alpha = 0.5), cv2.COLORMAP_JET)

    # AprilTag 감지
    color_image_with_tags = detect_apriltags(color_image)

    cv2.imshow('RGB with AprilTag', color_image_with_tags)
    cv2.imshow('Depth', depth_cm)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
cv2.destroyAllWindows()

