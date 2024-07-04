import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag
import rospy
from std_msgs.msg import Float32

def detect_apriltags(image, depth_frame):
    # AprilTag 검출기 생성
    detector = apriltag.Detector(options=apriltag.DetectorOptions(families='tag36h11'))

    # 회색조로 이미지 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # AprilTag 검출
    detections, dimg = detector.detect(gray, return_image=True)

    # 검출된 태그 주위에 사각형 그리기
    if len(detections) > 0:
        detected_centers = []  # 검출된 태그의 중심 좌표들을 저장할 리스트
        distances = []  # 검출된 태그의 거리들을 저장할 리스트

        for detection in detections:
            # Handle different types of detections (byte vs dictionary)
            if isinstance(detection, bytes):
                continue  # Skip if detection is not in expected format

            # 꼭지점 좌표 추출
            corners = detection.corners.astype(int)

            # 꼭지점 순서를 [lb, rb, rt, lt]로 정렬 (시계 방향)
            rect = np.array([corners[i] for i in [0, 1, 2, 3]], dtype=np.int32)

            # 사각형 그리기
            cv2.polylines(image, [rect], True, (0, 255, 0), 2)

            # 태그 번호 표시
            if detection.tag_id == 0:  # AprilTag의 ID가 0일 때만 처리
                cv2.putText(image, f"ID: {detection.tag_id}", (rect[0][0], rect[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # AprilTag의 중심 좌표 계산
                center_x = (rect[0][0] + rect[2][0]) // 2
                center_y = (rect[0][1] + rect[2][1]) // 2
                detected_centers.append((center_x, center_y))  # 중심 좌표 저장

                # 중심 좌표 출력
                cv2.putText(image, f"Center: ({center_x}, {center_y})", (rect[0][0], rect[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # AprilTag의 거리 계산
                distance = depth_frame.get_distance(center_x, center_y)
                distances.append(distance)  # 거리 저장
                cv2.putText(image, f"Distance: {distance:.2f} m", (rect[0][0], rect[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 검출된 AprilTag가 4개 이상일 때만 추가 정보 표시
        if len(detected_centers) >= 4:
            # 중심 좌표를 order_points 함수로 정렬
            detected_centers = order_points(np.array(detected_centers))

            # 4개의 태그 중심 좌표를 이어서 사각형을 그립니다.
            points = np.array(detected_centers[:4], dtype=np.int32)
            cv2.polylines(image, [points], True, (255, 0, 0), 2)

            # 사각형의 좌변과 우변 차이를 이용한 기울어진 정도 계산
            left_diff = np.linalg.norm(detected_centers[0] - detected_centers[3])
            right_diff = np.linalg.norm(detected_centers[1] - detected_centers[2])
            tilt_angle = np.arctan2(left_diff - right_diff, left_diff + right_diff) * 180 / np.pi
            cv2.putText(image, f"Tilt Angle: {tilt_angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 거리의 평균 계산
            avg_distance = np.mean(distances)
            cv2.putText(image, f"Avg Distance: {avg_distance:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 4개의 태그로 만들어진 사각형의 화면 중심으로부터의 거리 계산
            rect_center_x = np.mean(points[:, 0])
            rect_center_y = np.mean(points[:, 1])
            rect_distance_from_center = rect_center_x - image.shape[1] / 2
            cv2.putText(image, f"Rect Distance from Center: {rect_distance_from_center:.2f} pixels", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 토픽 발행
            publish_to_topic(tilt_angle, avg_distance, rect_distance_from_center)

    return image

def order_points(pts):
    # 좌표의 합을 기준으로 좌상단과 우하단 찾기
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="int")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 좌표의 차이를 기준으로 우상단과 좌하단 찾기
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def publish_to_topic(tilt_angle, avg_distance, rect_distance_from_center):
    # 토픽 생성
    tilt_angle_pub = rospy.Publisher('tilt_angle', Float32, queue_size=10)
    avg_distance_pub = rospy.Publisher('avg_distance', Float32, queue_size=10)
    rect_distance_from_center_pub = rospy.Publisher('rect_distance_from_center', Float32, queue_size=10)

    # 정보 발행
    tilt_angle_pub.publish(tilt_angle)
    avg_distance_pub.publish(avg_distance)
    rect_distance_from_center_pub.publish(rect_distance_from_center)

# ROS 노드 초기화
rospy.init_node('apriltag_info_publisher', anonymous=True)

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

while not rospy.is_shutdown():
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    # AprilTag 감지 및 정보 표시
    color_image_with_tags = detect_apriltags(color_image, depth_frame)

    #cv2.imshow('RGB with AprilTag', color_image_with_tags)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
cv2.destroyAllWindows()
