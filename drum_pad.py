import numpy as np

frame_width = 640
frame_height = 360


class DrumPad:
    def __init__(self, points, colour, midi_note):
        self.pts = points
        self.colour = colour
        self.midi_note = midi_note
        self.face_side = np.sign((points[1][0] - points[0][0]) * (0 - points[1][1]) -
                                 (points[1][1] - points[0][1]) * (0.5 - points[1][0]))

    def get_points(self):
        return self.pts

    def get_face_side(self):
        return self.face_side

    def hit(self):
        print(self.midi_note)
        return self.midi_note
