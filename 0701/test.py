import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, TransformStamped, Quaternion
from cv_bridge import CvBridge
import cv2
import numpy as np
import pyrealsense2 as rs
import tf2_ros
import tf
import tf_conversions

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector')
        self.image_pub = rospy.Publisher('aruco_image', Image, queue_size=10)
        self.pose_pub = rospy.Publisher('pose', Pose, queue_size=10)
        self.bridge = CvBridge()

        # RealSense 카메라 초기화
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.cfg)

        # 카메라 내부 파라미터 얻기
        profile = self.pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.cmtx = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        self.dist = np.array(intrinsics.coeffs)

        # ArUco 마커 설정
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()

        # 마커 사이즈 (mm 단위)
        self.marker_size = 65
        self.marker_3d_edges = np.array([
            [-self.marker_size / 2, -self.marker_size / 2, 0],
            [self.marker_size / 2, -self.marker_size / 2, 0],
            [self.marker_size / 2, self.marker_size / 2, 0],
            [-self.marker_size / 2, self.marker_size / 2, 0]
        ], dtype='float32')

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def timer_callback(self, event):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # 이미지를 numpy 배열로 변환
        img = np.asanyarray(color_frame.get_data())

        # ArUco 마커 검출
        corners, ids, rejected = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            for i, corner in enumerate(corners):
                corner = corner.reshape((4, 2))

                # PnP 문제 해결
                ret, rvec, tvec = cv2.solvePnP(self.marker_3d_edges, corner, self.cmtx, self.dist)
                if ret:
                    # 회전 벡터를 쿼터니언으로 변환
                    rotation_matrix = cv2.Rodrigues(rvec)[0]
                    quat = tf_conversions.transformations.quaternion_from_matrix(rotation_matrix)

                    pose_msg = Pose()
                    pose_msg.position.y = tvec[0][0] / 1000.0
                    pose_msg.position.x = -tvec[1][0] / 1000.0
                    pose_msg.position.z = tvec[2][0] / 1000.0
                    pose_msg.orientation = Quaternion(*quat)

                    # TF 브로드캐스트
                    t = TransformStamped()
                    t.header.stamp = rospy.Time.now()
                    t.header.frame_id = 'camera_link'
                    t.child_frame_id = f'aruco_marker_{ids[i][0]}'
                    t.transform.translation.y = tvec[0][0] / 1000.0
                    t.transform.translation.x = -tvec[1][0] / 1000.0
                    t.transform.translation.z = tvec[2][0] / 1000.0
                    t.transform.rotation = Quaternion(*quat)

                    self.tf_broadcaster.sendTransform(t)
                    self.pose_pub.publish(pose_msg)

        # 결과 이미지 ROS 메시지로 변환하여 퍼블리시
        img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.image_pub.publish(img_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ArucoDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass

