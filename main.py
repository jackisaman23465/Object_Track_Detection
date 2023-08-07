# This is a sample Python script.
import numpy as np
import cv2
import os
import function as fc
from PIL import Image
import math

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

folder = "image_folder/sample1/"
allfiles = os.listdir(folder)
imlist = [filename for filename in allfiles if filename[-4:] in [".jpg"]]
count = 0
line_list = []
thresh = 50
similar = 1

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    while (True):
        if count == len(imlist) - 1:
            break
        color_sim_list = []
        keep_list = []
        frame1_list = []
        frame2_list = []
        bg = Image.open(folder + imlist[0])
        img1 = Image.open(folder + imlist[count])
        img2 = Image.open(folder + imlist[count + 1])

        img_w, img_h = bg.size
        width = 500
        height = (int)((width / img_w) * img_h)
        bg = bg.resize((width, height), Image.BILINEAR)
        img1 = img1.resize((width, height), Image.BILINEAR)
        img2 = img2.resize((width, height), Image.BILINEAR)

        bg = np.array(bg)
        img1 = np.array(img1)
        img2 = np.array(img2)

        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        bg_grey = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # weight1 = bg_grey[0][0]/grey1[0][0]
        # weight2 = bg_grey[0][0]/grey2[0][0]
        # tr_grey1 = np.array((grey1*weight1),dtype=np.uint8)
        # tr_grey2 = np.array((grey2*weight2),dtype=np.uint8)

        bg_blur = cv2.GaussianBlur(bg_grey, (7, 7), 0)
        blur1 = cv2.GaussianBlur(grey1, (7, 7), 0)
        blur2 = cv2.GaussianBlur(grey2, (7, 7), 0)

        d1 = cv2.absdiff(blur1, bg_blur)
        d2 = cv2.absdiff(blur2, bg_blur)

        th1 = cv2.threshold(d1, thresh, 255, cv2.THRESH_BINARY)[1]
        th2 = cv2.threshold(d2, thresh, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((3, 3), np.uint8)

        dilated1 = cv2.dilate(th1, kernel, iterations=4)
        dilated2 = cv2.dilate(th2, kernel, iterations=4)

        contours1 = cv2.findContours(dilated1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours2 = cv2.findContours(dilated2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # cv2.drawContours(img2, contours2, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)

        if contours1 is not None:
            areas1 = [cv2.contourArea(c) for c in contours1]
        else:
            areas1 = []  # 將 areas1 設為空列表或其他適當的值

        if contours2 is not None:
            areas2 = [cv2.contourArea(c) for c in contours2]
        else:
            areas2 = []  # 將 areas2 設為空列表或其他適當的值

        for a in range(len(areas1)):
            if 2000 < areas1[a] < 6000:
                x, y, w, h = cv2.boundingRect(contours1[a])
                if w * h < 9000:
                    cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    frame1_list.append(
                        (fc.hist(img1[y:y + h, x:x + w, :]), ((int)((2 * x + w) / 2), (int)((2 * y + h) / 2))))

        for a in range(len(areas2)):
            if 2000 < areas2[a] < 6000:
                x, y, w, h = cv2.boundingRect(contours2[a])
                if w * h < 9000:
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    frame2_list.append((fc.hist(img2[y:y + h, x:x + w, :]), ((int)((2 * x + w) / 2), (int)((2 * y + h) / 2))))

        for i in range(len(frame1_list)):
            if len(frame1_list) == 0 or len(frame2_list) == 0:
                break
            color_sim = []
            first_color = 1
            for j in range(len(frame2_list)):
                norm = np.linalg.norm(frame1_list[i][0] - frame2_list[j][0])
                color_sim.append((norm, (i, j)))
            color_sim.sort()
            for j in range(len(color_sim_list)):
                if color_sim_list[j][1][1] == color_sim[0][1][1] and color_sim_list[j][0] > color_sim[0][0]:
                    color_sim_list[j] = color_sim[0]
                    first_color = 0
                    break
                if color_sim_list[j][1][1] == color_sim[0][1][1] and color_sim_list[j][0] < color_sim[0][0]:
                    print(color_sim)
                    del color_sim[0]
            if len(color_sim) != 0:
                if first_color:
                    color_sim_list.append((color_sim[0]))

        first_line = 1
        for i in range(len(color_sim_list)):
            if len(color_sim_list) == 0:
                break

            f1, f2 = (frame1_list[color_sim_list[i][1][0]][1], frame2_list[color_sim_list[i][1][1]][1])

            if sum(list(map(lambda x, y: (x - y) ** 2, f1, f2))) ** (1 / 2) < 300:
                for j in range(len(line_list)):
                    if f1 == line_list[j][-1][1]:
                        line_list[j].append([f1, f2])
                        keep_list.append(j)
                        first_line = 0
                        break
                    else:
                        first_line = 1
                if first_line:
                    line_list.append([[f1, f2]])
                    keep_list.append(len(line_list) - 1)

        for i in keep_list:
            for j in line_list[i]:
                cv2.line(img2, j[0], j[1], (0, 255, 0), 5)

        for i in keep_list:
            for j in line_list:
                if line_list.index(j) == i:
                    distance = sum(list(map(lambda x, y: (x - y) ** 2, j[0][0], j[-1][1]))) ** (1 / 2) * 30 / 500
                    m1 = (j[0][1][1] - j[0][0][1]) / (j[0][1][0] - j[0][0][0])
                    m2 = (j[-1][-1][1] - j[0][0][1]) / (j[-1][-1][0] - j[0][0][0])
                    angle = math.degrees(math.atan((m1 - m2) / (1 + m1 * m2)))
                    print("obj", line_list.index(j), "distance =", distance, "speed =", distance / len(j), "angle =", angle)

        cv2.imshow("frame4", th2)
        cv2.imshow("frame5", dilated2)
        cv2.imshow("frame6", img2)
        count += 1
        if cv2.waitKey(1000) == 13:
            continue
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
