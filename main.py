import time

import cv2
import numpy as np
import colorsys
import drumstick
import drum_pad
import keypoint
from midiBuffer import midiBuffer


# Frame dimensions
frame_width = 1280
frame_height = 720

# Set state to detect drumsticks
state = "detect_drumsticks"

# Set up the video feed
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Drumsticks
drumstick_1 = drumstick.Drumstick()
drumstick_2 = drumstick.Drumstick()

# Array to hold FPS values
fps = []

# Create Drum pads
drum_pads = []

kick = drum_pad.DrumPad(((500, 600), (780, 600)), (0, 0, 255), 36)
drum_pads.append(kick)

snare = drum_pad.DrumPad(((200, 500), (400, 600)), (0, 100, 255), 38)
drum_pads.append(snare)

closed_hihat = drum_pad.DrumPad(((1080, 500), (880, 600)), (0, 255, 255), 42)
drum_pads.append(closed_hihat)

'''
# Empty array to hold user interface
user_interface = np.zeros((frame_height, frame_width, 3), np.uint8)
#cv2.line(user_interface, kick.pts[0], kick.pts[1], kick.colour, 2)


for dp in drum_pads:
    cv2.line(user_interface, dp.pts[0], dp.pts[1], dp.colour, 2)
'''


def get_distance(pt1, pt2):
    return (((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2)) ** 0.5


def draw_drumpads(frame):
    for dp in drum_pads:
        cv2.line(frame, dp.pts[0], dp.pts[1], dp.colour, 2)


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
        detect_box_height = 25
        detect_box_width = 25

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

            # Draw detection box with border the same colour as the detected colour
            cv2.rectangle(frame, (detect_box_x1, detect_box_y1), (detect_box_x2, detect_box_y2), (b, g, r), 4)

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
        params.maxArea = 1280*720

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
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
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

            full_mask = mask_1 + mask_2

            # Erode and dilate to remove noise
            kernel = np.ones((5, 5), np.uint8)
            full_mask = cv2.erode(full_mask, kernel, iterations=1)
            kernel = np.ones((51, 51), np.uint8)
            full_mask = cv2.dilate(full_mask, kernel, iterations=1)

            # Detect blobs
            keypoints = detector.detect(full_mask)

            # Array to hold processed keypoints
            my_keypoints = []

            # Process the keypoints
            for kp in keypoints:
                my_keypoints.append(keypoint.Keypoint(kp.pt, kp.size))

            # Update tracked drumsticks
            if drumstick_1.tracked:
                drumstick_1.update(my_keypoints)
            if drumstick_2.tracked:
                drumstick_2.update(my_keypoints)

            # Find lost drumsticks
            if not drumstick_1.tracked:
                drumstick_1.find(my_keypoints)
            if not drumstick_2.tracked:
                drumstick_2.find(my_keypoints)

            # Check for hits
            #drumstick_1.check_for_hit(drum_pads)
            #drumstick_2.check_for_hit(drum_pads)

            mb.playChord([drumstick_1.check_for_hit(drum_pads)], 1, 48, onset=mb.getTime())
            mb.playChord([drumstick_2.check_for_hit(drum_pads)], 1, 48, onset=mb.getTime())

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
            draw_drumpads(frame)
            #frame = cv2.add(frame, user_interface)
            #frame = cv2.addWeighted(frame, 0.5, user_interface, 1, 1)

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
