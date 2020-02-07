import sys
import cv2
import imutils
import scipy as sci
import numpy as np
import matplotlib.pyplot as plt

class BlockDetector:
    def __init__(self):
        # color mapping:
        self.color_mapping = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (255, 255, 0),
            4: (0, 0, 255),
            5: (255, 0, 255),
            6: (0, 255, 255),
            7: (255, 255, 255)
        }
        # filt parameters
        self.rescale_ = 1.0
        # depth filt
        self.board_depth_ = 91.0  # cm
        self.sensor_range_ = 20.0
        self.depth_upper_bound_ = 1000
        self.depth_lower_bound_ = 200
        self.filt_gap_ = 3.5  # cm
        self.filt_threshold_ = 1.0  # cm
        # boundary filt
        self.plane_center_ = [260, 300]
        self.plane_size_ = 200
        self.machine_size_ = 30

        # box threshold
        self.box_size_thresh_ = 400
        self.box_ratio_thresh_ = 0.2  # abs(1 - length / width) < thresh
        # set color & make thresh
        self.color_ratio_threshold_ = 0.8
        # box overlap
        self.box_overlap_ratio_ = 0.5
        # data structure
        self.color_class_map_ = None
        self.boxes_ = []
        self.boxdepth_ = []
        self.modified_depth_map_ = None
        self.box_centers_ = []
        self.box_depth_ = []
        self.box_colors_ = []
        self.world_box_angles_ = []
        self.world_box_corners_ = []
        self.world_box_centers_ = []

    ## PreProcess
    def Preprocess(self, raw_img):
        # Nan
        new_img = np.copy(raw_img)
        new_img[np.isnan(new_img)] = 0
        # rescale img
        new_img[new_img > self.depth_upper_bound_] = 0
        new_img[new_img < self.depth_lower_bound_] = 0
        hist_stat = np.histogram(new_img.ravel(), bins=np.arange(1, self.depth_upper_bound_, 1), density=True)
        kinect_depth_index = np.argmax(hist_stat[0])
        kinect_board_depth = hist_stat[1][kinect_depth_index]
        self.rescale = self.board_depth_ / kinect_board_depth
        new_img = new_img.astype(np.float32)
        new_img *= self.rescale
        # change it starting from 0
        new_img[new_img >= (self.board_depth_ - self.filt_threshold_)] = 0
        new_img = self.board_depth_ - new_img
        # remove useless sensor part
        height_threshold = self.board_depth_ - self.sensor_range_
        new_img[new_img >= height_threshold] = 0
        return new_img

    def BoundaryFilter(self, depth_img):
        # Filter
        new_img = np.copy(depth_img)
        new_img[:(self.plane_center_[0] - self.plane_size_), :] *= 0
        new_img[(self.plane_center_[0] + self.plane_size_):, :] *= 0
        new_img[:, :(self.plane_center_[1] - self.plane_size_)] *= 0
        new_img[:, (self.plane_center_[1] + self.plane_size_):] *= 0

        # Center filter
        new_img[(self.plane_center_[0] - self.machine_size_):
                (self.plane_center_[0] + self.machine_size_),
                (self.plane_center_[1] - self.machine_size_):
                (self.plane_center_[1] + self.machine_size_)] *= 0
 
        return new_img

    def DepthFilter(self, raw_img):
        new_img = np.copy(raw_img)
        num_of_layer = int(new_img.max() / self.filt_gap_) + 1
        for i in range(1, num_of_layer + 1):
            if i == 1:
                low_bound = (i - 1) * self.filt_gap_
            else:
                low_bound = (i - 1) * self.filt_gap_ + self.filt_threshold_
            upper_bound = i * self.filt_gap_ - self.filt_threshold_
            assert(upper_bound > low_bound)
            new_img[np.logical_and(new_img < upper_bound, new_img > low_bound)] *= 0
        return new_img
    
    def Filter(self, depth_img, bgr_img):
        new_depth_img = np.copy(depth_img)
        new_bgr_img = np.copy(bgr_img)
        # filter depth img
        new_depth_img = self.Preprocess(depth_img)
        new_depth_img = self.BoundaryFilter(new_depth_img)
        # new_depth_img = self.DepthFilter(new_depth_img)

        self.modified_depth_map = new_depth_img
        new_bgr_img[new_depth_img == 0] = 0
        return new_depth_img, new_bgr_img
    
    def ColorClassify(self, pixel_color):
        b = pixel_color[0]
        g = pixel_color[1]
        r = pixel_color[2]
        dominate = max(b, max(g, r))
        if dominate == 0:
            return 0
        b /= dominate
        g /= dominate
        r /= dominate
        color_class = 0
        if r >= self.color_ratio_threshold_:
            color_class += 1
        if g >= self.color_ratio_threshold_:
            color_class += 2
        if b >= self.color_ratio_threshold_:
            color_class += 4
        return color_class

    def ColorMap(self, bgr_img):
        # caculate color class map
        self.color_class_map = np.zeros((bgr_img.shape[0], bgr_img.shape[1]), np.int8)
        aug_color_class_map = np.zeros(bgr_img.shape, np.int8)
        max_color_pixel = np.max(bgr_img, axis=-1)
        max_color_pixel = max_color_pixel[:, :, np.newaxis]
        color_ratio_pixel = bgr_img / max_color_pixel
        aug_color_class_map[
            color_ratio_pixel[:, :, 2] >= self.color_ratio_threshold_
        ] += 1
        aug_color_class_map[
            color_ratio_pixel[:, :, 1] >= self.color_ratio_threshold_
        ] += 2
        aug_color_class_map[
            color_ratio_pixel[:, :, 0] >= self.color_ratio_threshold_
        ] += 4
        self.color_class_map = aug_color_class_map[:, :, 0]

    ## Block Detection
    def Detect(self, bgr_img):
        # get gray image
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        # get color map
        self.ColorMap(bgr_img)

        # detect for each color
        for i in range(1, 8):
            masked_gray_img = np.copy(gray_img)
            masked_gray_img[self.color_class_map != i] *= 0
            # cv2.imshow("Masked Image {}".format(i), masked_gray_img)
            # cv2.waitKey(0)
            # contour
            _, contours, hierarchy = cv2.findContours(masked_gray_img, cv2.RETR_TREE,  cv2.CHAIN_APPROX_NONE)
            # contours, hierarchy = cv2.findContours(masked_gray_img, cv2.RETR_TREE,  cv2.CHAIN_APPROX_NONE)
            # find box
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box_size, box_ratio = self.BoxSize(box)
                if (box_size >= self.box_size_thresh_) and (abs(box_ratio - 1.0) < self.box_ratio_thresh_):
                    self.boxes_.append(box)
                    self.box_colors_.append(i)  # log the box color
        return
    
    def BoxSize(self, box_points):
        box_length = np.sqrt(np.square(box_points[0] - box_points[1]).sum())
        box_width = np.sqrt(np.square(box_points[2] - box_points[1]).sum())
        box_ratio = min(box_length, box_width) / (max(box_length, box_width) * 1.0)  ## All boxes should be square
        return box_length * box_width, box_ratio
    
    def BoxOverlap(self, box1, box2):
        axis_0 = (box1[1] - box1[0]).astype(np.float32)
        axis_0 /= np.linalg.norm(axis_0)
        axis_1 = (box1[2] - box1[1]).astype(np.float32)
        axis_1 /= np.linalg.norm(axis_1)
        axis_2 = (box2[1] - box2[0]).astype(np.float32)
        axis_2 /= np.linalg.norm(axis_2)
        axis_3 = (box2[2] - box2[1]).astype(np.float32)
        axis_3 /= np.linalg.norm(axis_3)
        axises = [axis_0, axis_1, axis_2, axis_3]
        overlap_ratio = 1
        for axis in axises:
            # projection for box1
            max_box1 = -np.inf
            min_box1 = np.inf
            max_box2 = -np.inf
            min_box2 = np.inf
            for point in box1:
                projection_value = point[0] * axis[1] + point[1] * axis[0]
                if projection_value > max_box1:
                    max_box1 = projection_value
                if projection_value < min_box1:
                    min_box1 = projection_value
            for point in box2:
                projection_value = point[0] * axis[1] + point[1] * axis[0]
                if projection_value > max_box2:
                    max_box2 = projection_value
                if projection_value < min_box2:
                    min_box2 = projection_value
            # overlap ratio in this axis
            if (max_box1 < min_box2) or (max_box2 < min_box1):
                return 0
            else:
                overlaped_length = min(max_box1 - min_box2, max_box2 - min_box1)
                overlap_ratio *= max(overlaped_length/(max_box1 - min_box1), 
                                      overlaped_length/(max_box2 - min_box2))
        return overlap_ratio

    def CheckOverlap(self):
        removed_boxes = dict()
        for i in range(len(self.boxes_)):
            removed_boxes[i] = False

        for i in range(len(self.boxes_)):
            if removed_boxes[i]:
                continue
            for j in range(i + 1, len(self.boxes_)):
                if removed_boxes[j]:
                    continue
                overlap_ratio = self.BoxOverlap(self.boxes_[i], self.boxes_[j])
                if overlap_ratio > self.box_overlap_ratio_:
                    # leave the same one
                    boxsize_i, _ = self.BoxSize(self.boxes_[i])
                    boxsize_j, _ = self.BoxSize(self.boxes_[j])
                    if boxsize_i < boxsize_j:  # remove the small one
                        removed_boxes[i] = True
                    else:
                        removed_boxes[j] = True
        new_boxes = list()
        new_box_colors = list()
        # find the colored one
        for i in range(len(self.boxes_)):
            if not (removed_boxes[i]):
                new_boxes.append(self.boxes_[i])
                new_box_colors.append(self.box_colors_[i])
        self.boxes_ = new_boxes
        self.box_colors_ = new_box_colors
        return

    def ShowBoxes(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        for color_index, box in zip(self.box_colors_, self.boxes_):
            color = self.color_mapping[color_index]
            rgb_img = cv2.polylines(rgb_img, [box], True, color)
        return rgb_img

    def BoxInfo(self):
        for box in self.boxes_:
            average_x = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
            average_y = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4.0
            # calculate the angle
            self.box_centers_.append((average_x, average_y))  # centers represented in x, y
            self.box_depth_.append(self.modified_depth_map[int(average_y), int(average_x)])
            # print("Box : Y: {}, X: {}, D: {}".format(average_y, average_x, self.box_depth[-1]))
        return
    
    def Process(self, depth_img, bgr_img):
        new_depth_img, new_bgr_img = self.Filter(depth_img, bgr_img)

        # print(new_depth_img.max())
        # plt.hist(new_depth_img.ravel(), 100)
        # plt.show()
        self.Detect(new_bgr_img)
        self.CheckOverlap()
        new_bgr_img = self.ShowBoxes(new_bgr_img)
        self.BoxInfo()
        # Draw boundary
        new_bgr_img = cv2.rectangle(new_bgr_img, (self.plane_center_[1] - self.plane_size_, 
                                                  self.plane_center_[0] - self.plane_size_),
                                                 (self.plane_center_[1] + self.plane_size_, 
                                                  self.plane_center_[0] + self.plane_size_),
                                                 (255, 0, 0), 2)
        new_bgr_img = cv2.rectangle(new_bgr_img, (self.plane_center_[1] - self.machine_size_, 
                                                  self.plane_center_[0] - self.machine_size_),
                                                 (self.plane_center_[1] + self.machine_size_, 
                                                  self.plane_center_[0] + self.machine_size_),
                                                 (255, 0, 0), 2)
        ## Save datas
        cv2.imwrite("./data/detected_img.png", new_bgr_img)
        rbg_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./data/raw_img.png", rbg_img)
        np.save("./data/depth.npy", depth_img)
        box_centers = np.array(self.box_centers_)
        np.save("./data/box_centers.npy", box_centers)
        
        return
    
    def Clear(self):
        # clean all data save structure
        self.color_class_map_ = None
        self.boxes_ = []
        self.boxdepth_ = []
        self.modified_depth_map_ = None
        self.box_centers_ = []
        self.box_depth_ = []
        self.box_colors_ = []
        self.world_box_angles_ = []
        self.world_box_corners_ = []
        self.world_box_centers_ = []


if __name__ == "__main__" :
    block_detector = BlockDetector()
    # depth_img = cv2.imread("data/depth.png")
    depth_img = np.load("data/depth.npy")
    bgr_img = cv2.imread("data/rgb.png")
    block_detector.Process(depth_img, bgr_img)
    
