import cv2


def draw_line(image, xf1, yf1, xf2, yf2):
    w = image.shape[1]
    h = image.shape[0]
    
    start_point = (int(w*xf1), int(h*yf1) )
    end_point = (int(w*xf2), int(h*yf2) )
    
    # Gets intercept
    slope = (yf2-yf1)/(xf2-xf1)
    b = yf1 - slope*xf1
    print("yf = " + str(round(slope, 3)) + "*xf + " + str(round(b,3)) )

    cv2.line(image, start_point, end_point, (255,0,0), 4)


def writes_area_text(image, text, xf1, yf1):
    w = image.shape[1]
    h = image.shape[0]
    
    start_point = (int(w*xf1), int(h*yf1) )
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255,100,100)
    thickness = 2
    
    # Draws background text
    cv2.putText(image, text, 
                start_point, font, fontScale, (0,0,0), thickness+3)
    
    # Draws foreground text
    cv2.putText(image, text, 
                start_point, font, fontScale, color, thickness)

image = cv2.imread("image.jpg")

w = image.shape[1]
h = image.shape[0]

draw_line(image, 0.00, 0.20, 0.10, 0.20) # Top-left line
draw_line(image, 0.10, 0.25, 0.55, 0.05) # Top-middle line
draw_line(image, 0.10, 0.25, 0.30, 0.80) # Left line
draw_line(image, 0.35, 0.15, 0.65, 0.45) # Middle Line
draw_line(image, 0.30, 0.80, 0.85, 0.25) # Bottom line
draw_line(image, 0.55, 0.05, 0.85, 0.25) # Right line

# X sections: 0.10, 0.30, 0.35, 0.55, 0.65, 0.85

# Writes text of each area
writes_area_text(image, "Register", 0.01, 0.25)
writes_area_text(image, "Area 2 (A2)", 0.20, 0.05)
writes_area_text(image, "Area 3 (A3)", 0.30, 0.40)
writes_area_text(image, "Entrance", 0.70, 0.80)
writes_area_text(image, "Area 1 (A1)", 0.60, 0.20)


cv2.imwrite("image-drawn.jpg", image)