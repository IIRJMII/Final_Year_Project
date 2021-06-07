import time
import cv2
import numpy as np
import colorsys
import drumstick
import drum_pad
from midiBuffer import midiBuffer

# Draw debugging information
debug = False

# Frame dimensions
frame_width = 640
frame_height = 360
frame_width_scaled = 1920
frame_height_scaled = 1080
scale_value = frame_width_scaled / frame_width

# Set state to register drumsticks
state = "register_drumsticks"

# Whether we are using the drum wands or not
drum_wands = True

# Controls the UI for customising Drumpads
drumpad_drawer_open = False
drumpad_to_draw = "kick"

# Set up the video feed
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Need to sleep before setting exposure time
time.sleep(2)
if drum_wands:
    cap.set(15, -8.0)
else:
    cap.set(15, 0)

# Drumsticks
drumstick_1 = drumstick.Drumstick()
drumstick_2 = drumstick.Drumstick()

# Array to hold FPS values
fps = []

# List to hold Drum pads
drum_pads = []
# List to hold Drum pad while it is being drawn [starting point, colour, midi note]
drawing_drum_pad = {"active": False, "p1": (0, 0), "p2": (0, 0), "colour": (34, 87, 255), "midi_note": 36}

# Create the default drum pads
kick = drum_pad.DrumPad(
    points=((0.4, 0.8), (0.6, 0.8)),
    colour=(34, 87, 255),
    midi_note=36)
drum_pads.append(kick)

snare = drum_pad.DrumPad(
    points=((0.1, 0.7), (0.3, 0.8)),
    colour=(0, 152, 255),
    midi_note=38)
drum_pads.append(snare)

closed_hihat = drum_pad.DrumPad(
    points=((0.7, 0.8), (0.9, 0.7)),
    colour=(7, 193, 255),
    midi_note=42)
drum_pads.append(closed_hihat)

open_hihat = drum_pad.DrumPad(
    points=((0.9, 0.2), (0.8, 0.6)),
    colour=(136, 150, 0),
    midi_note=46)
drum_pads.append(open_hihat)

crash = drum_pad.DrumPad(
    points=((0.1, 0.2), (0.2, 0.6)),
    colour=(181, 81, 63),
    midi_note=49)
drum_pads.append(crash)


# Returns the distance in pixels between two points
def get_distance(pt1, pt2):
    return (((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2)) ** 0.5


# Returns True if both points p1 and p2 are on the same side of the line l1 l2
def same_side(p1, p2, l1, l2):
    result_point_1 = np.sign((l2[0] - l1[0]) * (p1[1] - l2[1]) - (l2[1] - l1[1]) * (p1[0] - l2[0]))
    result_point_2 = np.sign((l2[0] - l1[0]) * (p2[1] - l2[1]) - (l2[1] - l1[1]) * (p2[0] - l2[0]))
    return result_point_1 == result_point_2


# Draws horizontally and vertically centered text
def draw_centered_text(img, xy, txt, scale, col):
    text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0]

    x = int(xy[0] - (text_size[0] / 2))
    y = int(xy[1] + (text_size[1] / 2))

    cv2.putText(img=img,
                text=txt,
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=scale,
                color=col,
                thickness=1,
                lineType=cv2.LINE_AA)


# Draws the drum pads
def draw_drumpads(img):
    (height, width, d) = img.shape
    for dp in drum_pads:
        cv2.line(
            img=img,
            pt1=(int(width*dp.pts[0][0]), int(height*dp.pts[0][1])),
            pt2=(int(width*dp.pts[1][0]), int(height*dp.pts[1][1])),
            color=dp.colour,
            thickness=5,
            lineType=cv2.LINE_AA)
    if drawing_drum_pad["active"]:
        cv2.line(
            img=img,
            pt1=(int(width*drawing_drum_pad["p1"][0]), int(height*drawing_drum_pad["p1"][1])),
            pt2=(int(width*drawing_drum_pad["p2"][0]), int(height*drawing_drum_pad["p2"][1])),
            color=drawing_drum_pad["colour"],
            thickness=5,
            lineType=cv2.LINE_AA)


def draw_UI(img):
    if state == "register_drumsticks":
        if drumstick_1.get_hsv() is None:
            draw_centered_text(
                img=img,
                xy=(int(frame_width_scaled * 0.5), int(frame_height_scaled * 0.25)),
                txt="Fill the square below with the coloured top of the first drumstick and press space.",
                scale=1.25,
                col=(0, 0, 0)
            )
        else:
            draw_centered_text(
                img=img,
                xy=(int(frame_width_scaled * 0.5), int(frame_height_scaled * 0.25)),
                txt="Now repeat the same process for the second drumstick.",
                scale=1.25,
                col=(0, 0, 0)
            )
    elif state == "track_drumsticks":
        if drumpad_drawer_open:
            # The background for the drum pad drawer
            cv2.rectangle(
                img=img,
                pt1=(0, 0),
                pt2=(frame_width_scaled, int(frame_height_scaled * 0.1)),
                color=(70, 70, 70),
                thickness=-1)
            # Kick button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (0.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (2.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "kick" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (0.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (2.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(34, 87, 255),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (1.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="Kick",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (1.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(36)",
                scale=0.75,
                col=(255, 255, 255))
            # Snare button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (2.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (4.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "snare" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (2.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (4.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(0, 152, 255),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (3.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="Snare",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (3.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(38)",
                scale=0.75,
                col=(255, 255, 255))
            # Closed Hi-Hat button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (4.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (6.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "closed_hihat" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (4.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (6.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(7, 193, 255),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (5.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="Closed Hi-Hat",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (5.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(42)",
                scale=0.75,
                col=(255, 255, 255))
            # Open Hi-Hat button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (6.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (8.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "open_hihat" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (6.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (8.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(136, 150, 0),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (7.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="Open Hi-Hat",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (7.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(46)",
                scale=0.75,
                col=(255, 255, 255))
            # Crash button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (8.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (10.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "crash" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (8.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (10.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(181, 81, 63),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (9.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="Crash",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (9.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(49)",
                scale=0.75,
                col=(255, 255, 255))
            # Low Tom button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (10.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (12.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "Low Tom" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (10.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (12.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(255, 80, 200),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (11.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="Low Tom",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (11.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(45)",
                scale=0.75,
                col=(255, 255, 255))
            # Mid Tom button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (12.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (14.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "Mid Tom" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (12.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (14.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(86, 172, 0),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (13.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="Mid Tom",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (13.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(47)",
                scale=0.75,
                col=(255, 255, 255))
            # High Tom button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (14.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (16.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "High Tom" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (14.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (16.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(102, 0, 206),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (15.5 / 20)), int(frame_height_scaled * 0.035)),
                txt="High Tom",
                scale=0.75,
                col=(255, 255, 255))
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (15.5 / 20)), int(frame_height_scaled * 0.07)),
                txt="(48)",
                scale=0.75,
                col=(255, 255, 255))
            # Eraser button
            cv2.rectangle(
                img=img,
                pt1=(int(frame_width_scaled * (16.6 / 20)), 0),
                pt2=(int(frame_width_scaled * (18.4 / 20)), int(frame_height_scaled * 0.1)),
                color=(80, 80, 80) if drumpad_to_draw == "eraser" else (70, 70, 70),
                thickness=-1)
            cv2.line(
                img=img,
                pt1=(int(frame_width_scaled * (16.8 / 20)), int(frame_height_scaled * 0.01)),
                pt2=(int(frame_width_scaled * (18.2 / 20)), int(frame_height_scaled * 0.01)),
                color=(157, 161, 245),
                thickness=3)
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (17.5 / 20)), int(frame_height_scaled * 0.05)),
                txt="Eraser",
                scale=1,
                col=(255, 255, 255))
            # Close drumpad drawer button
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (19.5 / 20)), int(frame_height_scaled * 0.05)),
                txt="Close",
                scale=1,
                col=(255, 255, 255))

        else:
            # Open drumpad drawer button
            draw_centered_text(
                img=frame,
                xy=(int(frame_width_scaled * (15.5 / 16)), int(frame_height_scaled * 0.05)),
                txt="Edit",
                scale=1,
                col=(0, 0, 0))


def mouse_click(event, x, y, flags, param):
    global drumpad_drawer_open, drumpad_to_draw
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drumpad_drawer_open:
            # If the click is on the open drumpad drawer button
            if y <= frame_height_scaled * 0.1 and x >= frame_width_scaled * (15/16):
                drumpad_drawer_open = True
        else:
            # If the click is on the drumpad drawer
            if y <= frame_height_scaled * 0.1:
                # If the click is on the close drumpad drawer button
                if x >= frame_width_scaled * (19 / 20):
                    drumpad_drawer_open = False
                    drawing_drum_pad["active"] = False
                # If the click is on the kick button
                elif frame_width_scaled * (0.7 / 20) <= x <= frame_width_scaled * (2.3 / 20):
                    drumpad_to_draw = "kick"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (34, 87, 255)
                    drawing_drum_pad["midi_note"] = 36
                # If the click is on the snare button
                elif frame_width_scaled * (2.7 / 20) <= x <= frame_width_scaled * (4.3 / 20):
                    drumpad_to_draw = "snare"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (0, 152, 255)
                    drawing_drum_pad["midi_note"] = 38
                # If the click is on the closed hi-hat button
                elif frame_width_scaled * (4.7 / 20) <= x <= frame_width_scaled * (6.3 / 20):
                    drumpad_to_draw = "closed_hihat"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (7, 193, 255)
                    drawing_drum_pad["midi_note"] = 42
                # If the click is on the open hi-hat button
                elif frame_width_scaled * (6.7 / 20) <= x <= frame_width_scaled * (8.3 / 20):
                    drumpad_to_draw = "open_hihat"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (136, 150, 0)
                    drawing_drum_pad["midi_note"] = 46
                # If the click is on the crash button
                elif frame_width_scaled * (8.7 / 20) <= x <= frame_width_scaled * (10.3 / 20):
                    drumpad_to_draw = "crash"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (181, 81, 63)
                    drawing_drum_pad["midi_note"] = 49
                # If the click is on the low tom button
                elif frame_width_scaled * (10.7 / 20) <= x <= frame_width_scaled * (12.3 / 20):
                    drumpad_to_draw = "Low Tom"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (255, 80, 200)
                    drawing_drum_pad["midi_note"] = 45
                # If the click is on the medium tom button
                elif frame_width_scaled * (12.7 / 20) <= x <= frame_width_scaled * (14.3 / 20):
                    drumpad_to_draw = "Mid Tom"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (86, 172, 0)
                    drawing_drum_pad["midi_note"] = 47
                # If the click is on the high tom button
                elif frame_width_scaled * (14.7 / 20) <= x <= frame_width_scaled * (16.3 / 20):
                    drumpad_to_draw = "High Tom"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (102, 0, 206)
                    drawing_drum_pad["midi_note"] = 48
                # If the click is on the eraser button
                elif frame_width_scaled * (16.7 / 20) <= x <= frame_width_scaled * (18.3 / 20):
                    drumpad_to_draw = "eraser"
                    drawing_drum_pad["active"] = False
                    drawing_drum_pad["colour"] = (157, 161, 245)
            # If the click is not on the drumpad drawer
            else:
                # If there is no drumpad currently being drawn, begin to draw one
                if not drawing_drum_pad["active"]:
                    drawing_drum_pad["active"] = True
                    drawing_drum_pad["p1"] = (x / frame_width_scaled, y / frame_height_scaled)
                    drawing_drum_pad["p2"] = drawing_drum_pad["p1"]
                # If there is a drumpad being drawn, create the drumpad
                else:
                    drawing_drum_pad["active"] = False

                    if drumpad_to_draw == "eraser":
                        for dp in drum_pads:
                            # If the eraser line intersects the drum pad, delete the drum pad
                            if not same_side(drawing_drum_pad["p1"], drawing_drum_pad["p2"], dp.pts[0], dp.pts[1]) and \
                                    not same_side(dp.pts[0], dp.pts[1], drawing_drum_pad["p1"], drawing_drum_pad["p2"]):
                                drum_pads.remove(dp)
                    else:
                        dp = drum_pad.DrumPad(
                            points=(drawing_drum_pad["p1"], drawing_drum_pad["p2"]),
                            colour=drawing_drum_pad["colour"],
                            midi_note=drawing_drum_pad["midi_note"])
                        drum_pads.append(dp)
    elif event == cv2.EVENT_MOUSEMOVE:
        if y > frame_height_scaled * 0.1:
            if drawing_drum_pad["active"]:
                drawing_drum_pad["p2"] = (x / frame_width_scaled, y / frame_height_scaled)


while True:
    if state == "exit":
        cap.set(15, 0)
        cap.release()
        cv2.destroyAllWindows()
        # Close the midi buffer if one has been created
        if mb is not None:
            mb.close()
        break

    if state == "register_drumsticks":
        # Detection box size
        detect_box_height = frame_height/30
        detect_box_width = frame_height/30

        # Detection box points
        detect_box_x1 = int((frame_width / 2) - (detect_box_height / 2))
        detect_box_x2 = int((frame_width / 2) + (detect_box_height / 2))
        detect_box_y1 = int((frame_height / 2) - (detect_box_height / 2))
        detect_box_y2 = int((frame_height / 2) + (detect_box_height / 2))

        drumstick_to_register = "drumstick_1"

        while state == "register_drumsticks":
            # Record time at the start of the loop
            start = time.time()

            # Read frame
            ret, frame = cap.read()

            # Flip frame to mirror image
            frame = cv2.flip(frame, 1)

            # Get the data inside the detection box and convert to HSV
            detect_box_data = frame[detect_box_y1:detect_box_y2, detect_box_x1:detect_box_x2]
            detect_box_data = cv2.cvtColor(detect_box_data, cv2.COLOR_BGR2HSV)

            # Calculate the average hue, saturation, and value
            hue = int(np.median(detect_box_data[:, :, 0]))
            sat = int(np.median(detect_box_data[:, :, 1]))
            val = int(np.median(detect_box_data[:, :, 2]))

            # Convert the HSV value to bgr to use for border
            rgb_normalised = colorsys.hsv_to_rgb(hue/180, sat/255, val/255)
            b = int(255 * rgb_normalised[2])
            g = int(255 * rgb_normalised[1])
            r = int(255 * rgb_normalised[0])

            # Create translucent white overlay with section missing for the detection region
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, detect_box_y1), (255, 255, 255), -1)
            cv2.rectangle(overlay, (0, 0), (detect_box_x1, frame_height), (255, 255, 255), -1)
            cv2.rectangle(overlay, (0, detect_box_y2), (frame_width, frame_height), (255, 255, 255), -1)
            cv2.rectangle(overlay, (detect_box_x2, 0), (frame_width, frame_height), (255, 255, 255), -1)

            # Add the overlay
            frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

            # Scale up frame
            frame = cv2.resize(frame, (1920, 1080))

            # Draw detection box with border the same colour as the detected colour
            cv2.rectangle(frame, (detect_box_x1 * 3, detect_box_y1 * 3),
                          (detect_box_x2 * 3, detect_box_y2 * 3), (b, g, r), 5)

            # Draw UI
            draw_UI(frame)

            cv2.imshow("Drums", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                state = "exit"
            elif key == ord(' '):
                if drumstick_to_register == "drumstick_1":
                    drumstick_1.set_hsv((hue, sat, val))
                    drumstick_to_register = "drumstick_2"
                else:
                    drumstick_2.set_hsv((hue, sat, val))
                    state = "track_drumsticks"

            # Record time at the end of the loop, calculate fps, then calculate average fps over the last 10 frames
            end = time.time()
            fps.append(1 / (end - start))
            if len(fps) > 10:
                fps.pop(0)
            print(int(sum(fps) / len(fps)), "fps")

    if state == "track_drumsticks":
        # Set the parameters for the blob detector
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = False
        params.minArea = 0
        params.maxArea = frame_width*frame_height

        params.filterByColor = False

        params.filterByInertia = False
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 1

        params.filterByConvexity = False
        params.minConvexity = 0
        params.maxConvexity = 1

        params.filterByCircularity = False
        params.minCircularity = 0
        params.maxCircularity = 1

        detector = cv2.SimpleBlobDetector_create(params)

        mb = midiBuffer(device=[], verbose=True)

        while state == "track_drumsticks":
            # Record the time at the start of the loop
            start = time.time()

            # Read frame
            ret, frame = cap.read()

            # Flip frame to mirror image
            frame = cv2.flip(frame, 1)

            # Blur to remove noise
            frame = cv2.GaussianBlur(frame, (int(frame_height/48), int(frame_height/48)), 0)

            # Convert to hsv
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Calculate mask for drumstick 1
            mask_bounds_1 = drumstick_1.get_mask_bounds()
            mask_1 = cv2.inRange(frame, np.array(mask_bounds_1[0][0]), np.array(mask_bounds_1[0][1]))

            if len(mask_bounds_1) > 1:
                mask_1 += cv2.inRange(frame, np.array(mask_bounds_1[1][0]), np.array(mask_bounds_1[1][1]))

            # Calculate mask for drumstick 2
            mask_bounds_2 = drumstick_2.get_mask_bounds()
            mask_2 = cv2.inRange(frame, np.array(mask_bounds_2[0][0]), np.array(mask_bounds_2[0][1]))

            if len(mask_bounds_2) > 1:
                mask_2 += cv2.inRange(frame, np.array(mask_bounds_2[1][0]), np.array(mask_bounds_2[1][1]))

            # Erode and dilate to remove noise
            kernel = np.ones(((int(frame_height / 144)), int(frame_height / 144)), np.uint8)
            mask_1 = cv2.erode(mask_1, kernel, iterations=1)
            mask_2 = cv2.erode(mask_2, kernel, iterations=1)

            kernel = np.ones((int(frame_height / 24), int(frame_height / 24)), np.uint8)
            mask_1 = cv2.dilate(mask_1, kernel, iterations=1)
            mask_2 = cv2.dilate(mask_2, kernel, iterations=1)

            # Detect blobs
            keypoints_mask_1 = detector.detect(mask_1)
            keypoints_mask_2 = detector.detect(mask_2)

            # Update drumsticks
            drumstick_1.update(keypoints_mask_1, drum_wands)
            drumstick_2.update(keypoints_mask_2, drum_wands)

            # Check for hits
            d1_hit = drumstick_1.check_for_hit(drum_pads, frame.shape[0], frame.shape[1])
            d2_hit = drumstick_2.check_for_hit(drum_pads, frame.shape[0], frame.shape[1])

            # Play drum hits
            mb.playChord(notes=d1_hit[0], dur=1, vel=d1_hit[1], onset=mb.getTime())
            mb.playChord(notes=d2_hit[0], dur=1, vel=d2_hit[1], onset=mb.getTime())

            # Convert frame back to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

            # Create translucent white overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (255, 255, 255), -1)

            # Add the overlay
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.1, 0)

            # Draw marker for drumstick if it is tracked
            if drumstick_1.tracked:
                cv2.circle(frame, tuple(drumstick_1.new_location), 10, (0, 255, 0), 10, cv2.LINE_AA)
                if debug:
                    # Line to represent velocity
                    cv2.line(frame, tuple(drumstick_1.new_location),
                             tuple(np.add(drumstick_1.new_location, drumstick_1.get_velocity())), (0, 200, 0), 2)
                    # Box to represent search area
                    s_a = drumstick_1.search_area
                    x1, x2, y1, y2 = int(s_a["min_x"]), int(s_a["max_x"]), int(s_a["min_y"]), int(s_a["max_y"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 0), 2)
            if drumstick_2.tracked:
                cv2.circle(frame, tuple(drumstick_2.new_location), 10, (0, 0, 255), 10, cv2.LINE_AA)
                if debug:
                    # Line to represent velocity for debugging purposes
                    cv2.line(frame, tuple(drumstick_2.new_location),
                             tuple(np.add(drumstick_2.new_location, drumstick_2.get_velocity())), (0, 0, 200), 2)
                    # Box to represent search area
                    s_a = drumstick_2.search_area
                    x1, x2, y1, y2 = int(s_a["min_x"]), int(s_a["max_x"]), int(s_a["min_y"]), int(s_a["max_y"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 150), 2)

            # Scale up frame
            frame = cv2.resize(frame, (1920, 1080))

            # Draw the drum pads
            draw_drumpads(frame)

            # Draw the UI
            draw_UI(frame)

            cv2.imshow("Drums", frame)
            cv2.setMouseCallback("Drums", mouse_click)

            key = cv2.waitKey(1)
            if key == ord('q'):
                state = "exit"

            # Record time at the end of the loop, calculate fps, then calculate average fps over the last 100 frames
            end = time.time()
            fps.append(1 / (end - start))
            if len(fps) > 10:
                fps.pop(0)
            print(int(sum(fps) / len(fps)), "fps")

    if state == "record_frames":
        frames = []
        masked_frames = []

        while state == "record_frames":
            # Read frame
            ret, frame = cap.read()

            # Flip frame to mirror image
            frame = cv2.flip(frame, 1)

            # Blur to remove noise
            frame = cv2.GaussianBlur(frame, (21, 21), 0)

            # Add to array
            frames.append(frame)

            # Convert to hsv
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Calculate mask for drumstick 1
            mask_bounds_1 = drumstick_1.get_mask_bounds()
            mask_1 = cv2.inRange(frame, np.array(mask_bounds_1[0][0]), np.array(mask_bounds_1[0][1]))

            if len(mask_bounds_1) > 1:
                mask_1 += cv2.inRange(frame, np.array(mask_bounds_1[1][0]), np.array(mask_bounds_1[1][1]))

            # Calculate mask for drumstick 2
            mask_bounds_2 = drumstick_2.get_mask_bounds()
            mask_2 = cv2.inRange(frame, np.array(mask_bounds_2[0][0]), np.array(mask_bounds_2[0][1]))

            if len(mask_bounds_2) > 1:
                mask_2 += cv2.inRange(frame, np.array(mask_bounds_2[1][0]), np.array(mask_bounds_2[1][1]))

            full_mask = mask_1 + mask_2

            # Add to array
            masked_frames.append(full_mask)

            frame = cv2.bitwise_and(frame, frame, mask=full_mask)

            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

            cv2.imshow("Recording frames", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                print(drumstick_1.get_mask_bounds())
                for i in range(len(frames)):
                    cv2.imwrite("recorded_frames/{}_A.png".format(i), frames[i])
                    cv2.imwrite("recorded_frames/{}_B.png".format(i), masked_frames[i])
                state = "exit"
