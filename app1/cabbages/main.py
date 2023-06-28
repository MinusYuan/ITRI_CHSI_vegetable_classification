import cv2
from yolo_module import yolov4

if __name__ == '__main__':
    image = cv2.imread('0311_RGB_10M_crop.png')
    n0, n1, n2, n3, draw_image = yolov4.predict(image)
    print(n0, n1, n2, n3)
    cv2.imwrite('/home/ken/yolo/test.jpg', draw_image)
