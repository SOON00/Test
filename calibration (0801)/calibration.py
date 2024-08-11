import numpy as np
import cv2
import pyrealsense2 as rs

# 체크보드 패턴의 행과 열 개수 (내부 코너 개수)
CHECKERBOARD = (6, 9)
# 준비할 3D 포인트와 2D 포인트 배열
objpoints = []  # 3D 포인트
imgpoints = []  # 2D 포인트

# 3D 포인트의 좌표 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)

# RealSense 카메라 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    count = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 컬러 프레임을 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(color_image, CHECKERBOARD, corners, ret)
            count += 1
            print(f"체크보드 이미지 수: {count}")

        cv2.imshow('RealSense', color_image)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("캘리브레이션 종료")
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    # 캘리브레이션 수행
    if objpoints and imgpoints:
        print(f"수집된 이미지 수: {len(objpoints)}")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            print("캘리브레이션 결과")
            print("카메라 행렬:\n", mtx)
            print("왜곡 계수:\n", dist)
        else:
            print("캘리브레이션 실패")
    else:
        print("체크보드 코너를 충분히 찾지 못했습니다.")
