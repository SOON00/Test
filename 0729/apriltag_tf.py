import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
from apriltag import apriltag
import transforms3d
import tf
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_matrix

class AprilTagTfBroadcaster:
    def __init__(self):
        rospy.init_node('apriltag_tf_broadcaster', anonymous=True)
    
        self.pose_publisher = rospy.Publisher('pose', Pose, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
    
        # RealSense 카메라 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
    
        # 카메라 내부 파라미터 얻기
        profile = self.pipeline.get_active_profile()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.cmtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs)
    
        # AprilTag 검출기 설정
        self.detector = apriltag("tag36h11")

        # 타이머 설정
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def detect_apriltags(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        
        if len(detections) > 0:
            for detection in detections:
                # 꼭지점 좌표 추출
                corners = np.array(detection['lb-rb-rt-lt'])
                
                # Convert to float32
                corners = corners.astype(np.float32)
                
                # Define 3D points of the tag's corners
                tag_size = 0.065  # Size of the tag in meters
                tag_corners_3d = np.array([
                    [-tag_size / 2, -tag_size / 2, 0],
                    [tag_size / 2, -tag_size / 2, 0],
                    [tag_size / 2, tag_size / 2, 0],
                    [-tag_size / 2, tag_size / 2, 0]
                ], dtype='float32')
                
                # Solve PnP to find the tag's pose
                ret, rvec, tvec = cv2.solvePnP(tag_corners_3d, corners, self.cmtx, self.dist)
                if ret:
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    # Convert rotation matrix to quaternion
                    quat = transforms3d.quaternions.mat2quat(rotation_matrix)

                    # Broadcast the transform
                    self.tf_broadcaster.sendTransform(
                        (tvec[0][0], tvec[1][0], tvec[2][0]),
                        (quat[1], quat[2], quat[3], quat[0]),
                        rospy.Time.now(),
                        f'apriltag_{detection["id"]}',
                        'camera_link'
                    )
                    
                    # Create and publish Pose message
                    pose_msg = Pose()
                    pose_msg.position.x = tvec[0][0]
                    pose_msg.position.y = tvec[1][0]
                    pose_msg.position.z = tvec[2][0]
                    pose_msg.orientation.x = quat[1]
                    pose_msg.orientation.y = quat[2]
                    pose_msg.orientation.z = quat[3]
                    pose_msg.orientation.w = quat[0]
                    self.pose_publisher.publish(pose_msg)

                    # Draw detected tag and its orientation on the image
                    self.draw_on_image(image, corners, tvec, rotation_matrix)
    
    def draw_on_image(self, image, corners, tvec, rotation_matrix):
        # Draw detected AprilTag
        corners = corners.astype(int)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        center = np.mean(corners, axis=0).astype(int)
        cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)

        # Draw orientation axes
        axis_length = 0.1  # Length of the axis in meters
        origin = (center[0], center[1])
        
        # Define the 3D axes
        axes = {
            'x': np.array([axis_length, 0, 0]),
            'y': np.array([0, axis_length, 0]),
            'z': np.array([0, 0, axis_length])
        }
        
        # Project 3D axes to 2D
        for axis_name, axis_vec in axes.items():
            axis_end = np.dot(rotation_matrix, axis_vec) + np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
            axis_end = np.dot(self.cmtx, axis_end[:3] / axis_end[2])
            axis_end = (int(axis_end[0]), int(axis_end[1]))
            
            color = {'x': (0, 0, 255), 'y': (0, 255, 0), 'z': (255, 0, 0)}[axis_name]
            cv2.line(image, origin, axis_end, color, 2)
            cv2.putText(image, axis_name, (axis_end[0], axis_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def timer_callback(self, event):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        
        img = np.asanyarray(color_frame.get_data())
        
        # Detect AprilTags and compute transforms
        self.detect_apriltags(img)

        # Display the image using OpenCV
        cv2.imshow('RGB with AprilTag', img)
        cv2.waitKey(1)

def main():
    apriltag_tf_broadcaster = AprilTagTfBroadcaster()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

