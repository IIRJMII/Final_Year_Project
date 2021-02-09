class Keypoint:
    def __init__(self, pt, size):
        self.pt = [int(pt[0]), int(pt[1])]
        self.size = size
        self.taken = False
