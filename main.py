import time

import cv2
import numpy as np
import colorsys
import drumstick
import drum_pad
import keypoint
from midiBuffer import midiBuffer


# Frame dimensions
frame_width = 640
frame_height = 360
frame_width_scaled = 1920
frame_height_scaled = 1080
scale_value = frame_width_scaled / frame_width

# Set state to detect drumsticks
state = "detect_drumsticks"

# Set up the video feed
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

time.sleep(2)

cap.set(15, -7.0)
#cap.set(15, -1.0)
#cap.set(15, 0)

# Drumsticks
drumstick_1 = drumstick.Drumstick()
drumstick_2 = drumstick.Drumstick()

# Array to hold FPS values
fps = []

# Create Drum pads
drum_pads = []

kick = drum_pad.DrumPad(
    points=((int(frame_width*0.4), int(frame_height*0.8)), (int(frame_width*0.6), int(frame_height*0.8))),
    colour=(0, 0, 255),
    midi_note=36)
drum_pads.append(kick)

snare = drum_pad.DrumPad(
    points=((int(frame_width*0.1), int(frame_height*0.7)), (int(frame_width*0.3), int(frame_height*0.8))),
    colour=(0, 100, 255),
    midi_note=38)
drum_pads.append(snare)

closed_hihat = drum_pad.DrumPad(
    points=((int(frame_width*0.7), int(frame_height*0.8)), (int(frame_width*0.9), int(frame_height*0.7))),
    colour=(0, 255, 255),
    midi_note=42)
drum_pads.append(closed_hihat)

open_hihat = drum_pad.DrumPad(
    points=((int(frame_width*0.7), int(frame_height*0.2)), (int(frame_width*0.9), int(frame_height*0.3))),
    colour=(0, 255, 0),
    midi_note=46)
#drum_pads.append(open_hihat)

crash = drum_pad.DrumPad(
    points=((int(frame_width*0.1), int(frame_height*0.3)), (int(frame_width*0.3), int(frame_height*0.2))),
    colour=(255, 0, 0),
    midi_note=49)
#drum_pads.append(crash)


def get_distance(pt1, pt2):
    return (((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2)) ** 0.5


def draw_drumpads(frame):
    for dp in drum_pads:
        cv2.line(frame, tuple([int(scale_value*p) for p in dp.pts[0]]), tuple([int(scale_value*p) for p in dp.pts[1]]),
                 dp.colour, 4)


while True:
    if state == "exit":
        cap.release()
        cv2.destroyAllWindows()
        # Close the midi buffer if one has been created
        if mb is not None:
            mb.close()
        break

    if state == "detect_drumsticks":
        # Detection box size
        detect_box_height = frame_height/30
        detect_box_width = frame_height/30


        # Detection box points
        detect_box_x1 = int((frame_width / 2) - (detect_box_height / 2))
        detect_box_x2 = int((frame_width / 2) + (detect_box_height / 2))
        detect_box_y1 = int((frame_height / 2) - (detect_box_height / 2))
        detect_box_y2 = int((frame_height / 2) + (detect_box_height / 2))

        drumstick_to_detect = "drumstick_1"

        while state == "detect_drumsticks":
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

            # Create white overlay with section missing for the detection region
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

            cv2.imshow("Detecting drumsticks", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                state = "exit"
            elif key == ord(' '):
                if drumstick_to_detect == "drumstick_1":
                    drumstick_1.set_hsv((hue, sat, val))
                    drumstick_to_detect = "drumstick_2"
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
            #frame = cv2.GaussianBlur(frame, (21, 21), 0)
            frame = cv2.GaussianBlur(frame, (int(frame_height/48), int(frame_height/48)), 0)
            #frame = cv2.fastNlMeansDenoisingColored(frame, h=1, templateWindowSize=3, searchWindowSize=5)

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
            # kernel = np.ones((5, 5), np.uint8)
            kernel = np.ones(((int(frame_height / 144)), int(frame_height / 144)), np.uint8)
            mask_1 = cv2.erode(mask_1, kernel, iterations=1)
            mask_2 = cv2.erode(mask_2, kernel, iterations=1)
            # kernel = np.ones((51, 51), np.uint8)
            kernel = np.ones((int(frame_height / 24), int(frame_height / 24)), np.uint8)
            mask_1 = cv2.dilate(mask_1, kernel, iterations=1)
            mask_2 = cv2.dilate(mask_2, kernel, iterations=1)

            # Detect blobs
            keypoints_mask_1 = detector.detect(mask_1)
            keypoints_mask_2 = detector.detect(mask_2)

            # Update drumsticks
            drumstick_1.update(keypoints_mask_1)
            drumstick_2.update(keypoints_mask_2)

            # Check for hits
            d1_hit = drumstick_1.check_for_hit(drum_pads)
            d2_hit = drumstick_2.check_for_hit(drum_pads)

            # Play drum hits
            mb.playChord(notes=[d1_hit[0]], dur=1, vel=d1_hit[1], onset=mb.getTime())
            mb.playChord(notes=[d2_hit[0]], dur=1, vel=d2_hit[1], onset=mb.getTime())

            '''
            frame = cv2.bitwise_and(frame, frame, mask=full_mask)

            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            '''

            frame = np.full_like(frame, 255)

            # Draw marker for drumstick if it is tracked
            if drumstick_1.tracked:
                cv2.circle(frame, tuple(drumstick_1.new_location), 10, (0, 255, 0), 10)
                # Line to represent velocity for debugging purposes
                cv2.line(frame, tuple(drumstick_1.new_location),
                         tuple(np.add(drumstick_1.new_location, drumstick_1.get_velocity())), (0, 200, 0), 2)
            if drumstick_2.tracked:
                cv2.circle(frame, tuple(drumstick_2.new_location), 10, (0, 0, 255), 10)
                # Line to represent velocity for debugging purposes
                cv2.line(frame, tuple(drumstick_2.new_location),
                         tuple(np.add(drumstick_2.new_location, drumstick_2.get_velocity())), (0, 0, 200), 2)

            # Draw the user interface
            #draw_drumpads(frame)
            #frame = cv2.add(frame, user_interface)
            #frame = cv2.addWeighted(frame, 0.5, user_interface, 1, 1)

            # Scale up frame
            frame = cv2.resize(frame, (1920, 1080))

            # Draw the drum pads
            draw_drumpads(frame)

            cv2.imshow("Detecting drumsticks", frame)

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
