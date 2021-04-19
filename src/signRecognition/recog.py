import random
import time
import cv2


def filter_regions(rects):
    x_array = []
    new_rects = []
    for i in range(len(rects)):
        if rects[i][0] not in x_array and rects[i][0] > 0 and rects[i][1] > 0 and \
                1100 < rects[i][2] * rects[i][3] < 2500 and \
                (rects[i][2] == rects[i][3] or rects[i][3] - rects[i][3] / 8 <= rects[i][2] <= rects[i][3] + rects[i][
                    3] / 8):
            new_rects.append(rects[i])
            x_array.append(rects[i][0])
    return new_rects


#
# image = cv2.imread("traficsings.jpg")
image = cv2.imread("tr2.png")
new = image[image.shape[0]:(int(image.shape[0] / 9)):-1, image.shape[1]:int(image.shape[1] / 2):-1]
new = new[image.shape[0]:int(image.shape[0] / 4):-1, image.shape[1]:0:-1]
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(new)
# ss.switchToSelectiveSearchQuality()
ss.switchToSelectiveSearchFast()
start = time.time()
rects = ss.process()
end = time.time()
print("[INFO] search took {:.4f} seconds".format(end - start))
rects = filter_regions(rects)
print("[INFO] {} total region proposals".format(len(rects)))
for (x, y, w, h) in rects:
    color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(new, (x, y), (x + w, y + h), color, 2)
cv2.imshow("Output", new)
key = cv2.waitKey(0) & 0xFF
