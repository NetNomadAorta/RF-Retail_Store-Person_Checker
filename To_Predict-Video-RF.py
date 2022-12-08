import os
import sys
import cv2
import time
import shutil
from PIL import Image
import io
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


# User parameters
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
MIN_SCORE               = 0.5
ROBOFLOW_MODEL          = "MODEL_NAME/MODEL_VERSION"
ROBOFLOW_API_KEY        = "API_KEY"


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


def make_appropriate_directory():
    if not os.path.exists(TO_PREDICT_PATH):
        os.makedirs(TO_PREDICT_PATH)
    if not os.path.exists(PREDICTED_PATH):
        os.makedirs(PREDICTED_PATH)


def draw_line(image, xf1, yf1, xf2, yf2):
    w = image.shape[1]
    h = image.shape[0]
    
    start_point = (int(w*xf1), int(h*yf1) )
    end_point = (int(w*xf2), int(h*yf2) )
    
    # # Gets intercept
    # slope = h*(yf2-yf1)/w*(xf2-xf1)
    # b = yf1 - slope*xf1
    # print(str(round(slope, 3)) + "*X + " + str(round(b,3)) )

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


def which_area(image, midx, midy):
    
    w = image.shape[1]
    h = image.shape[0]
    xf = midx/w
    yf = midy/h
    
    # x sections
    x1, x2, x3, x4, x5, x6 = 0.10, 0.30, 0.35, 0.55, 0.65, 0.85
    
    # y (mx+b) equations that separate each section
    y1 = 0.0*xf + 0.2 # Top-left line
    y2 = -0.444*xf + 0.294 # Top-middle line
    y3 = 2.75*xf + -0.025 # Left line
    y4 = -1.0*xf + 1.1 # Bottom line
    y5 = 1.0*xf + -0.2 # Middle Line
    
    if xf <= x1:
        if yf <= y1: # Top-left line
            area = "A2"
        else:
            area = "Register"
    elif xf > x1 and xf <= x2:
        if yf <= y2: # Top-middle line
            area = "A2"
        elif yf <= y3: # Left line
            area = "A3"
        else:
            area = "Register"
    elif xf > x2 and xf <= x3:
        if yf <= y2: # Top-middle line
            area = "A2"
        elif yf <= y4: # Bottom line
            area = "Area 3"
        else:
            area = "Entrance"
    elif xf > x3 and xf <= x4:
        if yf <= y2: # Top-middle line
            area = "A2"
        elif yf <= y5: # Middle Line
            area = "A1"
        elif yf <= y4: # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x4 and xf <= x5:
        if yf <= y5: # Middle Line
            area = "A1"
        elif yf <= y4: # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x5 and xf <= x6:
        if yf <= y4: # Bottom line
            area = "A1"
        else:
            area = "Entrance"
    else:
        area = "Entrance"
    
    return area


def object_match(coordinates, prev_coordinates):
    for prev_coordinate in prev_coordinates:
        if coordinates[-1][4] == prev_coordinate[4]:
            prev_x1 = prev_coordinate[0]
            prev_y1 = prev_coordinate[1]
            if (abs(x1-prev_x1) < frame_pixel_limiter
                and abs(y1-prev_y1) < frame_pixel_limiter
                ):
                coordinates[-1][5] = prev_coordinate[5] + 1/video_fps
    return coordinates[-1][5]


# Main()
# ==============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# If prediction folders don't exist, create them
make_appropriate_directory()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

# Start FPS timer
fps_start_time = time.time()

prev_coordinates = []
prev_prev_coordinates = []
prev_prev_prev_coordinates = []
ii = 0
# Goes through each video in TO_PREDICT_PATH
for video_name in os.listdir(TO_PREDICT_PATH):
    video_path = os.path.join(TO_PREDICT_PATH, video_name)
    
    video_capture = cv2.VideoCapture(video_path)
    
    # Video frame count and fps needed for VideoWriter settings
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = round( video_capture.get(cv2.CAP_PROP_FPS) )
    video_fps = int(video_fps)
    
    # If successful and image of frame
    success, image_b4_color = video_capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(PREDICTED_PATH + video_name, fourcc, video_fps, 
                                (int(image_b4_color.shape[1]), 
                                 int(image_b4_color.shape[0])
                                 )
                                )
    
    workers_in_frame_list = []
    count = 1
    while success:
        success, image_b4_color = video_capture.read()
        if not success:
            break
        
        # Inference through Roboflow section
        # -----------------------------------------------------------------------------
        # Load Image with PIL
        image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(image)
        
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage.save(buffered, quality=100, format="JPEG")
        
        # Construct the URL
        upload_url = "".join([
            "https://detect.roboflow.com/",
            ROBOFLOW_MODEL,
            "?api_key=",
            ROBOFLOW_API_KEY,
            "&confidence=",
            str(MIN_SCORE)
            # "&format=image",
            # "&stroke=5"
        ])
        
        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        
        response = requests.post(upload_url, 
                                 data=m, 
                                 headers={'Content-Type': m.content_type},
                                 )
        
        predictions = response.json()['predictions']
        # -----------------------------------------------------------------------------
        
        # Creates lines on image
        draw_line(image_b4_color, 0.00, 0.20, 0.10, 0.20) # Top-left line
        draw_line(image_b4_color, 0.10, 0.25, 0.55, 0.05) # Top-middle line
        draw_line(image_b4_color, 0.10, 0.25, 0.30, 0.80) # Left line
        draw_line(image_b4_color, 0.35, 0.15, 0.65, 0.45) # Middle Line
        draw_line(image_b4_color, 0.30, 0.80, 0.85, 0.25) # Bottom line
        draw_line(image_b4_color, 0.55, 0.05, 0.85, 0.25) # Right line
        
        # Creates lists from inferenced frames
        coordinates = []
        labels_found = []
        for prediction in predictions:
            x1 = int( prediction['x'] - prediction['width']/2 )
            y1 = int( prediction['y'] - prediction['height']/2 )
            x2 = int( prediction['x'] + prediction['width']/2 )
            y2 = int( prediction['y'] + prediction['height']/2 )
            
            coordinates.append([x1, y1, x2, y2, "section", 0])
            
            midx = int(prediction['x'])
            midy = int(prediction['y'])
            
            # Finds which area the coordinates belong to
            area = which_area(image_b4_color, midx, midy)
            coordinates[-1][4] = area
            
            # Checks to see if previous bounding boxes match with current ones
            #  If so, then adds to timer
            frame_pixel_limiter = 40
            coordinates[-1][5] = object_match(coordinates, prev_coordinates)
            
            # If didn't catch any matching, then checks 2 frames ago
            if coordinates[-1][5] == 0:
                coordinates[-1][5] = object_match(coordinates, prev_prev_coordinates)
            
            # If didn't catch any matching, then checks 3 frames ago
            if coordinates[-1][5] == 0:
                coordinates[-1][5] = object_match(coordinates, prev_prev_prev_coordinates)
            
            
            # Draws text above person of what area they are in
            text = area + ": {} sec".format(round(coordinates[-1][5],1))
            start_point = ( x1, max(y1-5, 20) )
            color = (125, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.60
            thickness = 2
            cv2.putText(image_b4_color, text, start_point, 
                        font, fontScale, (0,0,0), thickness+1)
            cv2.putText(image_b4_color, text, start_point, 
                        font, fontScale, color, thickness)
            
            # Draws bounding box
            cv2.rectangle(image_b4_color, (x1, y1), (x2, y2), color, 1)
            
        
        # Writes text of each area
        writes_area_text(image_b4_color, "Register", 0.01, 0.25)
        writes_area_text(image_b4_color, "Area 2 (A2)", 0.20, 0.05)
        writes_area_text(image_b4_color, "Area 3 (A3)", 0.30, 0.40)
        writes_area_text(image_b4_color, "Entrance", 0.70, 0.80)
        writes_area_text(image_b4_color, "Area 1 (A1)", 0.60, 0.20)
        
        
        # Saves current coordinates to previous for next frame
        prev_prev_prev_coordinates = prev_prev_coordinates.copy()
        prev_prev_coordinates = prev_coordinates.copy()
        prev_coordinates = coordinates.copy()
            
        
        # Saves video with bounding boxes
        video_out.write(image_b4_color)
        
        
        # Just prints out how fast video is inferencing and how much time left
        # ---------------------------------------------------------------------
        tenScale = 10
        ii += 1
        if ii % tenScale == 0:
            fps_end_time = time.time()
            fps_time_lapsed = fps_end_time - fps_start_time
            fps = round(tenScale/fps_time_lapsed, 2)
            time_left = round( (frame_count-ii)/fps )
            
            mins = time_left // 60
            sec = time_left % 60
            hours = mins // 60
            mins = mins % 60
            
            sys.stdout.write('\033[2K\033[1G')
            print("  " + str(ii) + " of " 
                  + str(frame_count), 
                  "-", fps, "FPS",
                  "-", "{}m:{}s".format(int(mins), round(sec) ),
                  end="\r", flush=True
                  )
            fps_start_time = time.time()
        # ---------------------------------------------------------------------
        
        count += 1
        # If you want to stop after so many frames to debug, uncomment below
        # if count == 300:
        #     break
    
    video_out.release()


print("\nDone!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)

# ==============================================================================

