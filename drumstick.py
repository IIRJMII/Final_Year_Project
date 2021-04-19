import time

import numpy as np

frame_width = 640
frame_height = 360
frame_center = (frame_width / 2, frame_height / 2)

#velocity_weights = [1/(2**i) for i in range(9, -1, -1)]


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
        self.velocities = [(0, 0) for i in range(10)]
        self.old_location = None
        self.new_location = None
        self.tracked = False
        self.frames_missing = 0

        self.test = 0

        self.search_area = {"min_x": 0,
                            "max_x": 0,
                            "min_y": 0,
                            "max_y": 0}

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

    def update(self, keypoints, drum_wands):
        # If the drumstick is currently being tracked
        if self.tracked:
            # Set the current old_location to the previous new_location
            self.old_location = self.new_location

            if not drum_wands:
                # Define a search area size, this should change with frames_missing
                search_area_size = frame_height/((self.frames_missing - 7.5) / -1.5)

                # Find the center of the search area by predicting where the drumstick is
                search_area_center = np.add(self.old_location, self.get_velocity())

                # Define the search area bounds
                self.search_area = {"min_x": search_area_center[0] - search_area_size,
                                    "max_x": search_area_center[0] + search_area_size,
                                    "min_y": search_area_center[1] - search_area_size,
                                    "max_y": search_area_center[1] + search_area_size}

                # Clamp the search area bounds
                self.search_area = {"min_x": 0 if self.search_area["min_x"] < 0 else self.search_area["min_x"],
                                    "max_x": frame_width if self.search_area["max_x"] > frame_width else self.search_area["max_x"],
                                    "min_y": 0 if self.search_area["min_y"] < 0 else self.search_area["min_y"],
                                    "max_y": frame_height if self.search_area["max_y"] > frame_height else self.search_area["max_y"]}
            else:
                # Define the search area bounds
                self.search_area = {"min_x": 0,
                                    "max_x": 1920,
                                    "min_y": 0,
                                    "max_y": 1080}

            # Initialise the best point to None
            best_point = None

            # Find the best point from the keypoints
            for kp in keypoints:
                # If the keypoint is within the search area
                if self.search_area["min_x"] <= kp.pt[0] <= self.search_area["max_x"] and \
                        self.search_area["min_y"] <= kp.pt[1] <= self.search_area["max_y"]:
                    if best_point is None:
                        best_point = kp
                    elif best_point.size < kp.size:
                        best_point = kp

            # If a valid point has been found
            if best_point is not None:
                self.new_location = [int(best_point.pt[0]), int(best_point.pt[1])]
                self.update_velocity(np.subtract(self.new_location, self.old_location))
                self.test = 0
                self.frames_missing = 0
            # If a valid point has not been found
            else:
                # Estimate the new location
                #self.new_location = np.add(self.old_location, self.get_velocity())

                print("Missing frames", self.frames_missing, time.time())
                self.frames_missing += 1
                # If the drumstick has been lost for more than 10 frames it is considered lost
                if self.frames_missing >= 10:
                    print("drumstick lost")
                    self.frames_missing = 0
                    self.tracked = False
        # Drumstick is not currently being tracked and needs to be found
        else:
            # Sort by size descending
            keypoints.sort(key=lambda x: x.size, reverse=True)

            if len(keypoints) > 0:
                kp = keypoints[0]
                self.old_location = [int(kp.pt[0]), int(kp.pt[1])]
                self.new_location = self.old_location
                self.update_velocity([0, 0])
                self.tracked = True

    def update_velocity(self, vel):
        self.velocities.append(vel)


        #if len(self.velocities) > 3:
        if len(self.velocities) > 10:
            self.velocities.pop(0)

    def get_velocity(self):
        x, y = zip(*self.velocities)
        weights = [10 / i for i in range(10, 0, -1)]
        return [int(np.average(a=x, axis=0, weights=weights)), int(np.average(a=y, axis=0, weights=weights))]
        #return [int(np.mean(x)), int(np.mean(y))]
        #return [int(np.average(a=x, axis=0, weights=velocity_weights)), int(np.average(a=y, axis=0, weights=velocity_weights))]

    def get_velocity_magnitude(self):
        x, y = self.get_velocity()
        #vel = (np.mean(x)**2 + np.mean(y)**2)**0.5
        vel = (x**2 + y**2)**0.5

        print("Vel:", vel)

        old_min, old_max = 0, 50
        new_min, new_max = 50, 100
        scaled_vel = int((((vel - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min)

        print("Scaled vel:", scaled_vel)

        '''
        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        scaled_vel = int((((vel - 0) * (127 - 0)) / (73 - 0)) + 0)
        if scaled_vel > 127:
            scaled_vel = 127
        '''

        #print(scaled_vel)
        return scaled_vel

    '''def check_for_hit(self, drum_pads):

        if self.old_location is None:
            return (-1, 0)

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
                        #return dp.hit()
                        return (dp.hit(), self.get_velocity_magnitude())
        return (-1, 0)'''

    def check_for_hit(self, drum_pads, height, width):

        if self.old_location is None:
            return (-1, 0)

        for dp in drum_pads:
            # Get the drum pad points
            dp_1, dp_2 = dp.get_points()
            dp_1 = (dp_1[0] * width, dp_1[1] * height)
            dp_2 = (dp_2[0] * width, dp_2[1] * height)
            # Check which side the drumstick is approaching the drum pad
            drumstick_side = np.sign((dp_2[0] - dp_1[0]) * (self.old_location[1] - dp_2[1]) -
                                     (dp_2[1] - dp_1[1]) * (self.old_location[0] - dp_2[0]))
            # If the drumstick is approaching the face of the drum pad
            if drumstick_side == dp.face_side:
                # Check if old_location and new_location are on different sides of the drum pad
                if not same_side(self.old_location, self.new_location, dp_1, dp_2):
                    # Check if drum pad points are on different sides of the line from old_location to new_location
                    if not same_side(dp_1, dp_2, self.old_location, self.new_location):
                        #return dp.hit()
                        return (dp.hit(), self.get_velocity_magnitude())
        return (-1, 0)
