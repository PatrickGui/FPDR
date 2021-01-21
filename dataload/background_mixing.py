from base_aug import *

class RandomChangeBack():
    def __init__(self, p=0.2, back_list_path='./background_img/back_list_6'):
        self.p = p
        self.back_list_path = back_list_path
    def __call__(self, img):
        if random.random() < self.p:
            back_list = []
            for v in os.listdir(self.back_list_path):
                back_list.append(os.path.join(self.back_list_path,v))
            i = int(np.floor(random.random()*len(back_list)))
            return self.changeBack(img, back_list[i])
        return img
    def changeBack(self, simg, timg):
        simg = cv2.cvtColor(np.asarray(simg), cv2.COLOR_RGB2BGR)
        # simg = cv2.imread(simg)
        timg = cv2.imread(timg)
        simg_hsv = cv2.cvtColor(simg, cv2.COLOR_BGR2HSV)
        # timg = cv2.resize(timg, None, fx=0.7, fy=0.7)
        # cv2.imshow('back', timg)
        mask = cv2.inRange(simg_hsv, np.array([0, 0, 0]), np.array([180, 255, 23]))  # vmax 46
        # cv2.imshow('mask', mask)
        # erode dilate
        erode = cv2.erode(mask, None, iterations=1)
        # cv2.imshow('erode', erode)
        dilate = cv2.dilate(erode, None, iterations=1)
        # cv2.imshow('dilate', dilate)
        # paste
        s = Image.fromarray(cv2.cvtColor(simg, cv2.COLOR_BGR2RGB))
        t = Image.fromarray(cv2.cvtColor(timg, cv2.COLOR_BGR2RGB))
        if s.size[0] > 256 or s.size[1] > 256:
            t = t.resize((s.size[0], s.size[1]))
        else:
            t = t.resize((256, 256))
        if s.size != t.size:
            print('img size != 3000')
        else:
            s.paste(t, None, Image.fromarray(dilate))
        return s
