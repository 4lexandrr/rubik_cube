import cv2
import numpy as np



class CubicRubik:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.colors = {}
        self.small_cube_coords = {}
    
        self.main()

    def draw_cube(self, image):

        board = 10
        cube_size = 50
        cube_padding = 5

        for row in range(3):
            for col in range(3):
                x1 =  col * (cube_size + cube_padding) + board
                y1 =  row * (cube_size + cube_padding) + board
                x2 = x1 + cube_size
                y2 = y1 + cube_size

                self.small_cube_coords[(row, col)] = [(y1, y2), (x1, x2)]

                small_cube = image[y1:y2, x1:x2]
                hsv_frame = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                self.detect_color(hsv_frame, small_cube)

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def main(self):

        while True:
            ret, frame = self.cap.read()
            flipped_frame = cv2.flip(frame, 1)

            self.draw_cube(flipped_frame)
        
            cv2.imshow('Frame', flipped_frame)

            k = cv2.waitKey(10)
            if k & 0xFF == ord('l'):
                print(self.small_cube_coords)

            if k & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


    def detect_cube(self, small_cube_rect):
        gray_image = cv2.cvtColor(small_cube_rect, cv2.COLOR_BGR2GRAY)
        blurred_gray_image = cv2.blur(gray_image, (3, 3))

        thresh = cv2.adaptiveThreshold(blurred_gray_image, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 0)

        contours, _ = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for i, contour in enumerate(contours):
            # Площадь одного кубика на стороне
            sq_one_cube = cv2.contourArea(contour)

            if sq_one_cube > 1000 and sq_one_cube < 8000:
                # Периметр одного кубика на стороне
                pm_one_cube = cv2.arcLength(contour, True)

                if cv2.norm(pm_one_cube**2/16, sq_one_cube) < 150:
                    x, y, w, h = cv2.boundingRect(contour)
                    x, y, w, h = x + 5, y + 5, w - 10, h - 10
                    cv2.rectangle(small_cube_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    def detect_color(self, hsv_frame, frame):
        red_lower = np.array([124, 60, 113], np.uint8)
        red_upper = np.array([174, 185, 146], np.uint8)
        red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

        green_lower = np.array([71, 134, 159], np.uint8)
        green_upper = np.array([74, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

        blue_lower = np.array([106, 71, 111], np.uint8)
        blue_upper = np.array([145, 140, 134], np.uint8)
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

        orange_lower = np.array([161, 113, 137], np.uint8)
        orange_upper = np.array([179, 255, 255], np.uint8)
        orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

        white_lower = np.array([97, 30, 144], np.uint8)
        white_upper = np.array([107, 50, 255], np.uint8)
        white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)

        yellow_lower = np.array([70, 118, 111], np.uint8)
        yellow_upper = np.array([88, 255, 147], np.uint8)
        yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)

        kernel = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(frame, frame, 
                                    mask = red_mask)   
        # For green color
        green_mask = cv2.dilate(green_mask, kernel)
        res_green = cv2.bitwise_and(frame, frame,
                                        mask = green_mask)
        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernel)
        res_blue = cv2.bitwise_and(frame, frame,
                                    mask = blue_mask)
        # For orange color
        orange_mask = cv2.dilate(orange_mask, kernel)
        res_orange = cv2.bitwise_and(frame, frame,
                                    mask = orange_mask)
        # For white color
        white_mask = cv2.dilate(white_mask, kernel)
        res_white = cv2.bitwise_and(frame, frame,
                                    mask = white_mask)
        # For yellow color
        yellow_mask = cv2.dilate(yellow_mask, kernel)
        res_yellow = cv2.bitwise_and(frame, frame,
                                    mask = yellow_mask)
        
        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(red_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 350):
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x + 5, y + 5), 
                                        (x + w - 5, y + h - 5), 
                                        (0, 0, 255), 2)
                
        # Creating contour to track green color
        contours, hierarchy = cv2.findContours(green_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 150):
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), 
                                        (x + w, y + h),
                                        (0, 255, 0), 2)
    
        # Creating contour to track blue color
        contours, hierarchy = cv2.findContours(blue_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 150):
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y),
                                        (x + w, y + h),
                                        (255, 0, 0), 2)
        
        # Creating contour to track orange color
        contours, hierarchy = cv2.findContours(orange_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 150):
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y),
                                        (x + w, y + h),
                                        (0, 191, 255), 2)
        
        # Creating contour to track white color
        contours, hierarchy = cv2.findContours(white_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 150):
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y),
                                        (x + w, y + h),
                                        (255, 255, 255), 2)
                
        # Creating contour to track yellow color
        contours, hierarchy = cv2.findContours(yellow_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 150):
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y),
                                        (x + w, y + h),
                                        (0, 255, 255), 2)


if __name__ == '__main__':
    App = CubicRubik()