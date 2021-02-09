import numpy as np

frame_height = 720
frame_width = 1280


class DrumPad:
    def __init__(self, points, colour, sound):
        self.pts = points
        self.colour = colour
        self.sound = sound
        self.face_side = np.sign((points[1][0] - points[0][0]) * ((frame_height / 2) - points[1][1]) -
                                 (points[1][1] - points[0][1]) * ((frame_width / 2) - points[1][0]))
        self.x = 0

    def get_points(self):
        return self.pts

    def get_face_side(self):
        return self.face_side

    def hit(self):
        print(self.x, self.sound)
        self.x += 1
