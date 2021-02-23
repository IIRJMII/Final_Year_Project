import numpy as np

frame_width = 1280
frame_height = 720
frame_center = (frame_width / 2, frame_height / 2)


def get_distance(pt1, pt2):
    return (((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2)) ** 0.5


def same_side(p1, p2, l1, l2):
    result_point_1 = np.sign((l2[0] - l1[0]) * (p1[1] - l2[1]) - (l2[1] - l1[1]) * (p1[0] - l2[0]))
    result_point_2 = np.sign((l2[0] - l1[0]) * (p2[1] - l2[1]) - (l2[1] - l1[1]) * (p2[0] - l2[0]))
    return result_point_1 == result_point_2


class Drumstick:
    def __init__(self):
        self.hsv = None
        self.mask_bounds = []
        self.velocities = []
        self.old_location = None
        self.new_location = None
        self.tracked = False
        self.frames_missing = 0

    def set_hsv(self, hsv):
        self.hsv = hsv
        self.calculate_mask_bounds(hsv)

    def get_hsv(self):
        return self.hsv

    def calculate_mask_bounds(self, hsv):
        #hue_var = 25
        #hue_var = 5
        hue_var = 15
        #sat_var = 100
        sat_var = 100
        #val_var = 100
        val_var = 100

        hue_lower = hsv[0] - hue_var
        hue_upper = hsv[0] + hue_var

        sat_lower = hsv[1] - sat_var
        if sat_lower < 0:
            sat_lower = 0
        sat_upper = hsv[1] + sat_var
        if sat_upper > 255:
            sat_upper = 255

        val_lower = hsv[2] - val_var
        if val_lower < 0:
            val_lower = 0
        val_upper = hsv[2] + val_var
        if val_upper > 255:
            val_upper = 255

        # If hue is below 0 or above 179, need to create two masks
        if hue_lower < 0:
            self.mask_bounds.append(((0, sat_lower, val_lower), (hue_upper, sat_upper, val_upper)))
            self.mask_bounds.append(((180 + hue_lower, sat_lower, val_lower), (179, sat_upper, val_upper)))
        elif hue_upper > 179:
            self.mask_bounds.append(((hue_lower, sat_lower, val_lower), (179, sat_upper, val_upper)))
            self.mask_bounds.append(((0, sat_lower, val_lower), (hue_upper - 180, sat_upper, val_upper)))
        else:
            self.mask_bounds.append(((hue_lower, sat_lower, val_lower), (hue_upper, sat_upper, val_upper)))
            print(self.mask_bounds)
            #self.mask_bounds.append(((hue_lower, 25, 140), (hue_upper, 255, 255)))
            #self.mask_bounds.append(((hue_lower, 50, 20), (hue_upper, 255, 255)))

    def get_mask_bounds(self):
        return self.mask_bounds

    def update(self, keypoints):
        # Set the current old_location to the previous new_location
        self.old_location = self.new_location

        # Define a search area size, this should change with frames_missing
        search_area_size = 150

        # Find the center of the search area by predicting where the drumstick is
        search_area_center = np.add(self.old_location, self.get_velocity())

        # Define the search area bounds
        search_area = {"min_x": search_area_center[0] - search_area_size,
                       "max_x": search_area_center[0] + search_area_size,
                       "min_y": search_area_center[1] - search_area_size,
                       "max_y": search_area_center[1] + search_area_size}

        # Clamp the search area bounds
        search_area = {"min_x": 0 if search_area["min_x"] < 0 else search_area["min_x"],
                       "max_x": 1280 if search_area["max_x"] > 1280 else search_area["max_x"],
                       "min_y": 0 if search_area["min_y"] < 0 else search_area["min_y"],
                       "max_y": 720 if search_area["max_y"] > 720 else search_area["max_y"]}

        # Initialise the best point to None
        best_point = None

        for kp in keypoints:
            # If the keypoints has not been taken
            if not kp.taken:
                # If the keypoint is within the search area
                if search_area["min_x"] <= kp.pt[0] <= search_area["max_x"] and \
                        search_area["min_y"] <= kp.pt[1] <= search_area["max_y"]:
                    if best_point is None:
                        best_point = kp
                    elif best_point.size < kp.size:
                        best_point = kp

        # If a valid point has been found
        if best_point is not None:
            self.new_location = best_point.pt
            self.update_velocity(np.subtract(self.new_location, self.old_location))
            best_point.taken = True
        # If a valid point has not been found
        else:
            # Estimate the new location
            #self.new_location = np.add(self.old_location, self.get_velocity())

            self.frames_missing += 1
            # If the drumstick has been lost for more than 10 frames it is considered lost
            if self.frames_missing == 10:
                print("drumstick lost")
                self.frames_missing = 0
                self.tracked = False

    def find(self, keypoints):
        # Sort by size descending
        keypoints.sort(key=lambda x: x.size, reverse=True)

        # Take the largest blob that has not been taken already
        for kp in keypoints:
            if not kp.taken:
                self.old_location = kp.pt
                self.new_location = kp.pt
                self.update_velocity([0, 0])
                self.tracked = True
                kp.taken = True
                break

    def update_velocity(self, vel):
        self.velocities.append(vel)

        if len(self.velocities) > 3:
            self.velocities.pop(0)

    def get_velocity(self):
        x, y = zip(*self.velocities)
        return [int(np.mean(x)), int(np.mean(y))]

    def check_for_hit(self, drum_pads):

        if self.old_location is None:
            return -1

        for dp in drum_pads:
            # Get the drum pad points
            dp_1, dp_2 = dp.get_points()
            # Check which side the drumstick is approaching the drum pad
            drumstick_side = np.sign((dp_2[0] - dp_1[0]) * (self.old_location[1] - dp_2[1]) -
                                     (dp_2[1] - dp_1[1]) * (self.old_location[0] - dp_2[0]))
            # If the drumstick is approaching the face of the drum pad
            if drumstick_side == dp.face_side:
                # Check if old_location and new_location are on different sides of the drum pad
                if not same_side(self.old_location, self.new_location, dp_1, dp_2):
                    # Check if drum pad points are on different sides of the line from old_location to new_location
                    if not same_side(dp_1, dp_2, self.old_location, self.new_location):
                        return dp.hit()
        return -1
