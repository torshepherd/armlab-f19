import kinematics
import numpy as np
import math


class TaskPlanner:
    def __init__(self, task):
        # predefined values
        self.block_height_ = 33
        self.identical_threshold_ = 1000  # distance threshold
        self.lift_distance_ = 30
        self.pick_distance_ = 18
        self.push_distance_ = 10
        hold_T = kinematics.FK_dh(np.array([0, 0, 0, 0, 0, 0]),6)
        self.hold_position_ = np.append(hold_T.t, hold_T.get_euler_angles())
        self.predefined_place_position_ = np.array([186, 0, 0,  -np.pi, 0, 0])
        # place plan: 0, 1
        # 0 : a pre-defined place position
        # 1 : a user-defined place position
        self.task_ = task
        # data storage structure
        # position is a 4d location (x, y, z, theta)
        self.place_position_ = self.predefined_place_position_
        self.place_floor_ = 0  # height of current tower
        self.height_map_ = {
            1 : 33.0,
            2 : 70.0007701,
            3 : 105.982281,
            4 : 152.896345
        }
        self.predefined_pose = {
            4: np.array([1.81893597e+02, 3.80334021e+00, 1.66539237e+02, 3.13788307e+00, -2.09407911e-01, 3.52962487e-02]),
            5: np.array([1.71039168e+02,  -5.73653219e+00,   1.85103733e+02,  -1.60585479e+00, -3.37168728e-02, -1.38722553e+00]),
            6: np.array([161.66916141, 4.95778355, 232.33232539, 0.31385444, -1.4608904, 2.85541216]),
            7: np.array([1.60035059e+02, 2.54526534e+00, 2.72297329e+02, -4.94056433e-03, -1.50092926e+00, -3.12223585e+00]),
            8: np.array([157.60025003, 3.33747212, 303.84739213, 0.94417065, -1.51085512, 2.19926289])
        }
        # color part:
        self.color_enable_ = False
        '''
        0 : white
        1 : red
        2 : green
        3 : yellow
        4 : blue
        5 : purple
        6 : light blue
        7 : black
        '''
        self.predefined_color_order = [1, 2, 3, 4, 5, 6, 7]
        # push mode:
        self.push_mode_ = False

    def Pick(self, box_centers, box_angles, box_colors):
        print(box_centers)
        # box centers are respresented in (x, y, z)
        # choose the nearest one to pick up
        index = -1
        place_index = -1
        min_distance = np.inf
        place_x = self.place_position_[0]
        place_y = self.place_position_[1]

        # find the height of placing place
        self.place_floor_ = 1  # default zero
        for i, box_center in enumerate(box_centers):
            distance = math.pow((place_x - box_center[0]), 2) + math.pow((place_y - box_center[1]), 2)
            print("Distance is {}".format(distance))
            if  (distance < min_distance) and (distance > self.identical_threshold_):
                min_distance = distance
                index = i
            elif distance <= self.identical_threshold_:
                # update the place_floor
                current_floor = np.round(box_center[2] / 30.0)
                self.place_floor_ = current_floor + 1
                print("Place Block Found! Height : {}".format(current_floor))
        
        # choose the nearest one & color
        if self.color_enable_:
            for i, color in enumerate(box_colors):
                if i is not place_index:
                    if color == self.predefined_color_order[self.place_floor_]:
                        index = i                        

        # from box angle to euler angle
        valid_euler = kinematics.IK_valid(box_centers[index], box_angles[index])
        if valid_euler is None:
            print("Warning!!!! Invalid block position")
            valid_euler = np.array([-np.pi, 0, 0])         
        # valid_euler = np.array([-np.pi, 0, 0])
        # Generate Waypoints for pick:
        if self.push_mode_:
            waypoints = self.Push(box_centers[index], valid_euler)
        else:
            box_height = self.Height(box_centers[index][2])
            waypoints = [
                    np.hstack([self.hold_position_, [0]]),
                    np.array([box_centers[index][0], box_centers[index][1], 
                        box_height + self.pick_distance_, valid_euler[0], valid_euler[1], valid_euler[2], 0]),
                    np.array([box_centers[index][0], box_centers[index][1], 
                        box_height, valid_euler[0], valid_euler[1], valid_euler[2], 0] ),
                    np.array([box_centers[index][0], box_centers[index][1], 
                        box_height, valid_euler[0], valid_euler[1], valid_euler[2], 1] ),
                    np.array([0.25]),
                    np.array([box_centers[index][0], box_centers[index][1], 
                        box_height + self.pick_distance_, valid_euler[0], valid_euler[1], valid_euler[2], 1]),
                    np.hstack([self.hold_position_, [0]])
                ]
        
        return waypoints
            
    def Place(self):
        # change place position
        if self.place_floor_ <= 3:
            self.place_position_[2] = self.height_map_[self.place_floor_] * 1.1
        else:
            self.place_position_ = self.predefined_pose[self.place_floor_]
        # generate the waypoints
        waypoints = [
                np.hstack([self.place_position_, 1]),      
                np.hstack([self.place_position_, 0]),  
                np.hstack([.2]),      
                np.hstack([self.place_position_ + np.array([0, 0, self.lift_distance_/2, 0, 0, 0]), 0]),
                np.hstack([2]),      
                np.hstack([self.hold_position_, 0])
            ]
        return waypoints
    
    def Push(self, box_center, valid_euler):
        box_height = self.Height(box_center[2])
        waypoints = [
                np.hstack([self.hold_position_, [0]]),
                np.array([box_center[0], box_center[1], 
                    box_height + self.lift_distance_, valid_euler[0], valid_euler[1], valid_euler[2], 0]),
                np.array([box_center[0], box_center[1], 
                    box_height, valid_euler[0], valid_euler[1], valid_euler[2], 0] ),
                np.array([box_center[0], box_center[1], 
                    box_height, valid_euler[0], valid_euler[1], valid_euler[2], 1] ),
                np.array([0.5]),
                np.array([box_center[0] + self.push_distance_, box_center[1], 
                    box_height, valid_euler[0], valid_euler[1], valid_euler[2], 1] ),
                np.array([box_center[0] + 2.0 * self.push_distance_, box_center[1], 
                    box_height, valid_euler[0], valid_euler[1], valid_euler[2], 1] ),
                np.array([box_center[0] + 3.0 * self.push_distance_, box_center[1], 
                    box_height, valid_euler[0], valid_euler[1], valid_euler[2], 1] ),
                np.array([box_center[0] + 3.0 * self.push_distance_, box_center[1], 
                    box_height, valid_euler[0], valid_euler[1], valid_euler[2], 0] ), 
                np.hstack([self.hold_position_, [0]])   
            ]
        return waypoints


    def Height(self, z):
        # height of block
        height = self.height_map_[np.round(z / self.block_height_)] * (z / self.block_height_)
        # height mayget tuned
        return height
    
    def Quit(self):
        # Clean everything
        self.place_position_ = np.zeros((6))
        self.place_floor_ = 0  # height of current tower
