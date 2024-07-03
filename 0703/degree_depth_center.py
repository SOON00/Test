import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32  # 추가: Float32 메시지 타입 가져오기
from cv_bridge import CvBridge
from apriltag import apriltag

class AprilTagDetector:
    def __init__(self):
        rospy.init_node('april_tag_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=10)
        self.orientation_pub = rospy.Publisher("/april_tag/orientation", Float32, queue_size=10)  # 수정: Float32 메시지 타입 사용
        self.distance_pub = rospy.Publisher("/april_tag/average_distance", Float32, queue_size=10)  # 수정: Float32 메시지 타입 사용
        self.center_offset_pub = rospy.Publisher("/april_tag/center_offset", Float32, queue_size=10)  # 추가: 중심 좌표 오프셋 토픽

        # RealSense 파이프라인 설정
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipe.start(self.cfg)

    def detect_apriltags(self, image, depth_frame):
        # AprilTag 검출기 생성
        detector = apriltag("tag36h11")

        # 회색조로 이미지 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # AprilTag 검출
        detections = detector.detect(gray)

        centers = []  # 중심 좌표를 저장할 리스트
        distances = []  # 거리를 저장할 리스트

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

                    # AprilTag의 거리 계산
                    distance = depth_frame.get_distance(center_x, center_y)
                    distances.append(distance)  # 거리 저장
                    # 거리 출력
                    cv2.putText(image, f"Distance: {distance:.2f} m", (rect[0][0], rect[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 중심 좌표가 4개인 경우 사각형 그리기 및 방향 계산
        if len(centers) == 4:
            centers = np.array(centers, dtype=np.int32)  # numpy 배열로 변환
            centers = self.order_points(centers)
            cv2.polylines(image, [centers], True, (0, 0, 255), 2)  # 빨간색 사각형 그리기
            
            # 평균 거리 계산
            if len(distances) > 0:
                average_distance = np.mean(distances)
                cv2.putText(image, f"Average Distance: {average_distance:.2f} m", (centers[0][0], centers[0][1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self.distance_pub.publish(average_distance)

            # orientation 계산
            orientation = self.calculate_orientation(centers)
            cv2.putText(image, f"Orientation: {orientation:.2f} degrees", (centers[0][0], centers[0][1] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.orientation_pub.publish(orientation)

            # 화면 중앙으로부터 중심의 오프셋 계산 및 출력
            image_center_x = image.shape[1] // 2
            tag_center_x = np.mean(centers[:, 0])
            offset_x = tag_center_x - image_center_x
            cv2.putText(image, f"Offset: {offset_x:.2f}", (centers[0][0], centers[0][1] - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.center_offset_pub.publish(offset_x)

        return image

    def order_points(self, pts):
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

    def calculate_orientation(self, centers):
        # 중심 좌표에서 첫 번째와 두 번째 포인트 사이의 벡터를 계산
        vector = centers[1] - centers[0]
        
        # 벡터의 각도를 계산
        angle = np.arctan2(vector[1], vector[0])
        
        # 각도를 도 단위로 변환
        angle_degrees = np.degrees(angle)
        
        return angle_degrees

    def run(self):
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            # 프레임 캡처
            frames = self.pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # 컬러 이미지 가져오기
            color_image = np.asanyarray(color_frame.get_data())

            # AprilTag 감지 및 평균 거리, 방향 계산
            color_image_with_tags = self.detect_apriltags(color_image, depth_frame)

            # 이미지 표시
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(color_image_with_tags, "bgr8"))
            cv2.imshow("AprilTag Detection", color_image_with_tags)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = AprilTagDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass

