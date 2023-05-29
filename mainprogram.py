import cv2
import numpy as np


def main(camera=0):
    cap = cv2.VideoCapture(camera)

    while True:
        ret, frame = cap.read()
        flipped_frame = cv2.flip(frame, 1)
        image, grid = detect_cube(flipped_frame)

        cv2.imshow('Frame', image)
        print(grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def find_color()
    pass


def detect_cube(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray_image = cv2.blur(gray_image, (3, 3))

    thresh = cv2.adaptiveThreshold(blurred_gray_image, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 0)

    contours, _ = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    grid_cube = []

    for contour in contours:
        # Площадь одного кубика на стороне
        sq_one_cube = cv2.contourArea(contour)

        if sq_one_cube > 1000 and sq_one_cube < 8000:
            # Периметр одного кубика на стороне
            pm_one_cube = cv2.arcLength(contour, True)

            if cv2.norm(pm_one_cube**2/16 - sq_one_cube) < 200:
                x, y, w, h = cv2.boundingRect(contour)
                x, y, w, h = x + 5, y + 5, w - 10, h - 10
                cube = np.array(cv2.mean(image[y:y+h,x:x+w])).astype(int)[:-1]
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                val = (50 * y) + (10 * x)
                cube = np.append(cube, val)
                grid_cube.append(cube) 

    if len(grid_cube) > 0:
        grid_cube = np.asarray(grid_cube)
        grid_cube = grid_cube[grid_cube[:, -1].argsort()]
    return image, grid_cube



if __name__ == '__main__':
    main(camera=0)