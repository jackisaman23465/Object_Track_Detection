import numpy as np
import cv2

def hist(img):
    bin = 50
    imgArr = np.array(img, dtype=float)

    imgArr_R = imgArr[:, :, 0].ravel()
    h_r = np.histogram(imgArr_R, bin, range=(0, 255))

    imgArr_G = imgArr[:, :, 1].ravel()
    h_g = np.histogram(imgArr_G, bin, range=(0, 255))

    imgArr_B = imgArr[:, :, 2].ravel()
    h_b = np.histogram(imgArr_B, bin, range=(0, 255))

    all_h_y = []
    all_h_y.append(h_r[0])
    all_h_y.append(h_g[0])
    all_h_y.append(h_b[0])

    all_h_y_Arr = np.array(all_h_y, dtype=float).ravel()
    all_h_y_Arr = all_h_y_Arr / np.sum(all_h_y_Arr)
    return all_h_y_Arr


def find_center_and_hist(frame_list, areas, contours, img):
    for a in range(len(areas)):
        if areas[a] > 2000 and areas[a] < 6000:
            x, y, w, h = cv2.boundingRect(contours[a])
            if w * h < 9000:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                frame_list.append((hist(img[y:y + h, x:x + w, :]), ((int)((2 * x + w) / 2), (int)((2 * y + h) / 2))))


def find_sim(sim_list, frame1_list, frame2_list, similar):
    for i in range(len(frame1_list)):
        if len(frame1_list) == 0 or len(frame2_list) == 0:
            break
        sim = 1
        sim_i = 0
        sim_j = 0
        for j in range(len(frame2_list)):
            norm = np.linalg.norm(frame1_list[i][0] - frame2_list[j][0])
            if sim > norm and norm < similar:
                sim = norm
                sim_i = i
                sim_j = j
        if np.linalg.norm(frame1_list[sim_i][0] - frame2_list[sim_j][0]) < similar:
            sim_list.append((sim_i, sim_j))


def find_trace(line_list, keep_list, frame1_list, frame2_list, sim_list):
    first_line = 1
    for i in range(len(sim_list)):
        if len(sim_list) == 0:
            break

        f1, f2 = (frame1_list[sim_list[i][0]][1], frame2_list[sim_list[i][1]][1])

        if sum(list(map(lambda x, y: (x - y) ** 2, f1, f2))) ** (1 / 2) < 500:
            for j in range(len(line_list)):
                if frame1_list[sim_list[i][0]][1] == line_list[j][-1][1]:
                    line_list[j].append([f1, f2])
                    keep_list.append(j)
                    first_line = 0
                    break
                else:
                    first_line = 1
            if first_line:
                line_list.append([[f1, f2]])
                keep_list.append(len(line_list) - 1)


def draw_trace(img, line_list, keep_list):
    for i in range(len(keep_list)):
        for j in range(len(line_list[keep_list[i]])):
            cv2.line(img, line_list[keep_list[i]][j][0], line_list[keep_list[i]][j][1], (0, 255, 0), 5)
