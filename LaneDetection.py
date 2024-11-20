import cv2
import numpy as np

import numpy as np

def average_lines(lines, threshold_slope):
    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
       
        
        if abs(slope) > threshold_slope:
            if slope < 0:
                left_lines.append([x1, y1, x2, y2])
            else:
                right_lines.append([x1, y1, x2, y2])

    def avg_line(lines):
        return np.mean(lines, axis=0, dtype=np.int32) if lines else None

    return avg_line(left_lines), avg_line(right_lines)

def draw_lines(img, line, color=(0, 0, 255), thickness=10):
    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def get_direction(left_line, right_line, width):
    if left_line is not None and right_line is not None:
        mid_x = (left_line[0] + right_line[2]) 
        if mid_x < width :
            return "Move left"
        elif mid_x > 2 * width:
            return "Move right"
        else:
            return "Go straight"
    elif left_line is not None:
        return "Move left"
    elif right_line is not None:
        return "Move right"
    else:
        return "No lane detected"


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([[(0, height), (width / 2, height / 2 + 50), (width, height)]], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, minLineLength=50, maxLineGap=200)
    if lines is not None:
        left_line, right_line = average_lines(lines, threshold_slope=0.5)
        line_image = np.zeros_like(frame)
        draw_lines(line_image, left_line)
        draw_lines(line_image, right_line)
        direction = get_direction(left_line, right_line, width)
        cv2.putText(line_image, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return frame

def main():
    cap = cv2.VideoCapture('curved_lane.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Lane and Direction Detection', process_frame(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

