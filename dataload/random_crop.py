from base_aug import *

class RandomCropping:
    def __init__(self, size=(448,448), interpolation=Image.BILINEAR, scale = (0.5,1.0), probability=0.75):
        self.size = size
        self.interpolation = interpolation
        self.min_area = scale[0]
        self.max_area = scale[1]
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img.resize(self.size, self.interpolation)
        scale = random.uniform(self.min_area, self.max_area)
        img = img.resize((int(np.ceil(self.size[0] * scale)), int(np.ceil(self.size[1] * scale))), self.interpolation)
        img = np.array(img)

        start = int((self.size[0] - img.shape[0]) / 2)
        mask = np.zeros((self.size[0], self.size[1], 3), np.float32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                mask[i + start, j + start, :] = img[i, j, :]
        mask = Image.fromarray(np.uint8(mask))
        return mask