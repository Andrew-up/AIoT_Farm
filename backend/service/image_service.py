import cv2 as cv
import numpy as np

kernel = np.ones((10, 10), np.uint8)

def main():
    video = cv.VideoCapture('../data/video/test.mp4')
    while True:
        ret, frame = video.read()
        image = frame
        if ret:
            # image = cv.imread('../data/image/test.png')
            h, w = image.shape[1], image.shape[0]
            image_resize = cv.resize(image, (h // 1, w // 1), cv.INTER_AREA)
            gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
            gaus = cv.GaussianBlur(gray, (7, 7), cv.BORDER_DEFAULT)
            ret, thresh = cv.threshold(gaus, 170, 255, cv.THRESH_BINARY)
            dilation = cv.erode(thresh, kernel, iterations=5)
            edged = cv.Canny(dilation, 10, 200)
            contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # cv.drawContours(image_resize, contours, -1, (0, 255, 0), 4)
            for contour in contours:
                (x, y, w, h) = cv.boundingRect(contour)
                cv.rectangle(image_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow('image', image_resize)
            # cv.waitKey(0)
            # print("test")
        if cv.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
    pass
