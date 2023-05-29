import cv2
import numpy as np


def find_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([40, 150, 20])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    cv2.imshow('mask', mask)
    return mask


def draw_cube(image, colors):

    for row in range(3):
        for col in range(3):
            x1 = col * 30
            y1 = row * 30
            x2 = x1 + 30
            y2 = y1 + 30

            if colors[row][col] == 255:
                color = (0, 0, 255)
            else:
                color =  (255, 255, 255)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)


def detect_cube(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray_image = cv2.blur(gray_image, (3, 3))

    thresh = cv2.adaptiveThreshold(blurred_gray_image, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 0)

    contours, _ = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    colors = []

    for contour in contours:
        # Площадь одного кубика на стороне
        sq_one_cube = cv2.contourArea(contour)

        if sq_one_cube > 1000 and sq_one_cube < 8000:
            # Периметр одного кубика на стороне
            pm_one_cube = cv2.arcLength(contour, True)

            if cv2.norm(pm_one_cube**2/16 - sq_one_cube) < 150:
                x, y, w, h = cv2.boundingRect(contour)
                x, y, w, h = x + 5, y + 5, w - 10, h - 10
                cube = np.array(cv2.mean(image[y:y + h, x:x + w])).astype(int)[:-1]
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                val = (50 * y) + (10 * x)
                colors.append(cube)
        
    return image, colors


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        flipped_frame = cv2.flip(frame, 1)
        image, grid = detect_cube(flipped_frame)
        mask = find_color(image)
        draw_cube(image, mask)

        cv2.imshow('Frame', image)
        
        if cv2.waitKey(1) & 0xFF == ord('l'):
            print(grid)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()