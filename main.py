import numpy as np
import cv2
import matplotlib.pyplot as plt

cube = np.array([[10, 0, 0, 1],[20,0,0,1],[20,5,0,1],[10,5,0,1],[10,0,5,1],
                [20,0,5,1],[20,5,5,1],[10,5,5,1]])
surface = np.array([[8,0,0,1],[8,5,0,1],[15,5,0,1],[15,0,0,1]])
chair = np.array([])


def find_calibration_matrix(img_points, resize_gray):
    # Arrays to store object points and image points from all the images.

    # 3d point in real world space

    #todo: bottle soap
    #obj_points = [[8,0,0], [8,6,0],[0,6,0],[8,6,12], [0,6,12], [8,0,12]]

    obj_points = [[0,5.2,0],[7.8,5.2,0],[7.8,0,0],[0,5.2,3.3],[7.8,5.2,
                                                               3.3],[7.8,0,
                                                                     3.3]]
    obj_points = np.array(obj_points)
    obj_points = obj_points.astype('float32')

    # 2d points in image plane.
    img_points = np.array(img_points)
    img_points = img_points.astype('float32')

    # calibration, rotation and translation
    camera_matrix = cv2.initCameraMatrix2D([obj_points], [img_points], resize_gray.shape[::-1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([obj_points],
                                                       [img_points],
                                                       resize_gray.shape[::-1],
                                                       camera_matrix,
                                                       None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    rvecs = np.array(rvecs).reshape(3,1)
    R, _ = cv2.Rodrigues(rvecs)
    Rt_matrix = np.concatenate((R, np.array(tvecs).reshape(3,1)), axis=1)

    # projection matrix
    P = camera_matrix @ Rt_matrix

    paint_object(P, resize_gray)


def paint_object(P, resize_gray):
    pts = []
    for point in surface:
    #point = np.array([7.8,0,0,1])
        point_to_draw = np.dot(P, point.reshape(4, 1))
        x = point_to_draw[0]/point_to_draw[2]
        y = point_to_draw[1]/point_to_draw[2]
        pts.append([x,y])
    pts = np.array(pts, np.int32).reshape((-1,1,2))
    resize_gray = cv2.polylines(resize_gray,[pts],True,(0, 255, 255))
    #resize_gray = cv2.circle(resize_gray, (x, y), 3, (255, 255, 255), 3)
    plt.imshow(resize_gray, cmap='gray')
    plt.show()


def click_event(event, x, y, flags, params):
    """
    collecting the points clicked on the image
    :param event: left mouse clicks
    :param x: x coordinate
    :param y: y coordinate
    """
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        if len(points) == 6:
            resize_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            find_calibration_matrix(points, resize_gray)
            # close the window
            cv2.destroyAllWindows()


def resize_image(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    points = []
    # reading the image
    img = cv2.imread("img3.jpeg", 1)

    resized = resize_image(img, 70)

    # displaying the image
    cv2.imshow('image', resized)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)








