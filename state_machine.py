import task_planner
import block_detect
import kinematics

import time
import numpy as np
import cv2
import math

"""
TODO: Add states and state functions to this class
        to implement all of the required logic for the armlab
"""


class StateMachine():

    def __init__(self, rexarm, planner, kinect):
        self.rexarm = rexarm
        self.tp = planner
        self.kinect = kinect
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypointlist = []
        # For Detection & Taskplanning
        self.block_detector = block_detect.BlockDetector()
        self.task_planner = task_planner.TaskPlanner(0)  # defaulty, we are using gripper 0
        self.mouse_pose = np.zeros(3)
        self.tlist = []
    
    def set_next_state(self, state):
        self.next_state = state

    """ This function is run continuously in a thread"""

    def run(self):
        if(self.current_state == "manual"):
            if (self.next_state == "manual"):
                self.manual()
            if(self.next_state == "idle"):
                self.idle()
            if(self.next_state == "estop"):
                self.estop()

        if(self.current_state == "idle"):
            if(self.next_state == "manual"):
                self.manual()
            if(self.next_state == "idle"):
                self.idle()
            if(self.next_state == "estop"):
                self.estop()
            if(self.next_state == "calibrate"):
                self.calibrate()
            if(self.next_state == "execute"):
                print('moved into execute')
                self.execute()
            if(self.next_state == "wait4record"):
                self.waypointlist = []
                self.wait4record()
            if(self.next_state == "wait4action"):
                self.wait4action()

        if(self.current_state == "estop"):
            self.next_state = "estop"
            self.estop()

        if(self.current_state == "calibrate"):
            if(self.next_state == "idle"):
                self.idle()

        if(self.current_state == "execute"):
            if(self.next_state == "estop"):
                self.estop()
            if(self.next_state == "idle"):
                self.idle()

        if(self.current_state == "wait4record"):
            if(self.next_state == "record"):
                self.record()
            elif(self.next_state == "idle"):
                self.rexarm.enable_torque()
                self.idle()
            elif(self.next_state == "wait4record"):
                self.wait4record()

        if(self.current_state == "record"):
            if(self.next_state == "wait4record"):
                self.wait4record()
        
        # pick & place relevant state
        if(self.current_state == "wait4action"):
            if(self.next_state == "wait4action"):
                self.wait4action()
            elif(self.next_state == "pick"):
                # self.pick()
                self.task1()
            elif(self.next_state == "place"):
                self.place()
            elif(self.next_state == "detection"):
                self.detection()
            elif(self.next_state == "quittask"):
                self.quittask()
        
        # Deal with other cases
        if(self.current_state == "detection"):
            if(self.next_state == "wait4action"):
                print("Quit Detection")
                self.wait4action()
        
        if(self.current_state == "pick"):
            if(self.next_state == "wait4action"):
                print("Quit Pick")
                self.wait4action()
        
        if(self.current_state == "place"):
            if(self.next_state == "wait4action"):
                print("Quit Place")
                self.wait4action()
        
        if(self.current_state == "quittask"):
            if(self.next_state == "idle"):
                print("QuitTask")
                self.idle()
                
        # print(self.current_state, self.next_state)
        
    """Functions run for each state"""
    def wait4action(self):
        # Stay at a specific place to wait for further action
        self.current_state = "wait4action"
        self.next_state = "wait4action"
        # Some control policy
        # If it is the first time, we should find a place to place the block
    
    def task1(self):
        while(self.task_planner.place_floor_ is not 8):
            self.detection()
            self.pick()
            self.place()
        
    def pick(self):
        # Choose a nearest block, move to there, grasp
        print("Start Picking")
        self.current_state = "pick"
        self.next_state = "wait4action"
        # Some control policy

        # Get waypoints
        # Based on detection
        block_centers = self.block_detector.world_box_centers_;
        block_angles = self.block_detector.world_box_angles_;
        block_colors = self.block_detector.box_colors_;
        # Based on click
        world_click = self.kinect.Pixel2World(
            self.kinect.last_click[0],
            self.kinect.last_click[1])
        
        # find the nearest one to click
        index = -1
        min_distance = np.inf
        for i, box_center in enumerate(block_centers):
            distance = math.pow((world_click[0] - box_center[0]), 2) + math.pow((world_click[1] - box_center[1]), 2)
            if  (distance < min_distance):
                min_distance = distance
                index = i
        print("The One we are picking : {}, {}".format(block_centers[index], block_angles[index]))
        block_centers = [block_centers[index]]
        block_angles = [block_angles[index]]
        block_colors = [block_colors[index]]
        
        waypoints = self.task_planner.Pick(block_centers, 
                                           block_angles,
                                           block_colors)
        # Feed waypoints into trajplanner
        print(waypoints)
        self.tp.execute_plan(waypoints, controller="IK")

    def place(self):
        # Move to the target place, release
        # After placing, we should check the height of results
        print("Start Placing")
        self.current_state = "place"
        self.next_state = "wait4action"
        # Get waypoints
        waypoints = self.task_planner.Place()
        # Feed waypoints into trajplanner
        print(waypoints)
        self.tp.execute_plan(waypoints, controller="IK")
        self.tp.plot_traj()

    def detection(self):
        print("Start Detection")
        # Detection the position of block in img coordinates
        self.current_state = "detection"
        self.next_state = "wait4action"
        # run on dectection
        self.block_detector.Clear()
        self.block_detector.Process(
            self.kinect.currentDepthFrame,
            self.kinect.currentVideoFrame)
        
        print("Finished This part")
        # transform the results from img coordinates to world coordinates
        self.block_detector.world_box_centers_ = []
        for box_center, box in zip(self.block_detector.box_centers_, self.block_detector.boxes_):
            u = box_center[0]
            v = box_center[1]
            coord = self.kinect.Pixel2World(u, v)
            self.block_detector.world_box_centers_.append(coord)
            print("Box : {}".format(box))
            # get angle from boxes
            u1 = box[0][0]
            v1 = box[0][1]
            u2 = box[1][0]
            v2 = box[1][1]
            coord1 = self.kinect.Pixel2World(u1, v1)
            coord2 = self.kinect.Pixel2World(u2, v2)
            box_angle = np.arctan2(float(coord2[1] - coord1[1]), float(coord2[0] - coord1[0]))
            print("Box angle : {}".format(box_angle))
            # make it to proper zone
            if(box_angle < 0):
                box_angle += np.pi
            if(box_angle > np.pi / 2.0):
                box_angle -= (np.pi / 2.0)
            self.block_detector.world_box_angles_.append(box_angle)
            print("Box : X: {}, Y: {}, D: {}, A: {}".format(coord[0], coord[1], coord[2], box_angle))
            
        print(self.block_detector.world_box_centers_)
        print("Finished This part2")
        return

    def quittask(self):
        print("Task Quit")
        # Quit block stacking task & clear relevant storage
        self.current_state = "quittask"
        self.next_state = "idle"
        # Clear relevant storage
        self.block_detector.Clear()

    def wait4record(self):
        self.current_state = "wait4record"
        self.next_state = "wait4record"
        self.rexarm.disable_torque()
        self.rexarm.get_feedback()

    def record(self):
        self.current_state = "record"
        self.next_state = "wait4record"
        joints = list(self.rexarm.joint_angles_fb)
        print(joints)
        self.waypointlist.append(joints)
        print(self.waypointlist)
        self.rexarm.disable_torque()
        self.rexarm.get_feedback()

    def execute(self):
        self.status_message = "State: Executing movements..."
        # Old execute functionality
        '''
        for pos in self.waypointlist:
            self.rexarm.set_speeds_normalized_global(.1)
            self.rexarm.set_positions(pos)
            self.rexarm.pause(3)
        '''
        # New functionality:
        '''
        testlist = [np.array([ 0.0, 0.0, 0.0, 0.0, 0.0]),
                    np.array([ 1.0, 0.8, 1.0, 0.5, 1.0]),
                    np.array([-1.0,-0.8,-1.0,-0.5, -1.0]),
                    np.array([-1.0, 0.8, 1.0, 0.5, 1.0]),
                    np.array([1.0, -0.8,-1.0,-0.5, -1.0]),
                    np.array([ 0.0, 0.0, 0.0, 0.0, 0.0])]
        '''
        testlist = []
        T_0 = kinematics.FK_dh(np.array([-1.58824929, 0., 1.39626311, 0., 0.785398, 0.]), 6)
        T_1 = kinematics.FK_dh(np.array([ 0,0.27925262,0.59341182,0,0,-0.80285129]),6)
        T_2 = kinematics.FK_dh(np.array([-1.34390324, 0.75049142, 1.30899667, 0., 0.29670591, 0.]), 6)
        T_3 = kinematics.FK_dh(np.array([-1.36135653, 0.71558484, 1.570796, 0., 0.03490658, 0.]), 6)
        T_4 = kinematics.FK_dh(np.array([0, 0, 0, 0, 0, 0]),6)
        T_5 = kinematics.FK_dh(np.array([0., 0.97738418, 1.06465062, 0, 0.3141592, -0.89011773]),6)
        print("====================================")
        print(np.append(T_0.t, T_0.get_euler_angles()))
        T_6 = kinematics.FK_dh(np.array([0, 0, 0, 0, 0, 0]),6)
        testlist.append(np.append(np.append(T_1.t, T_1.get_euler_angles()), 0))
        testlist.append(np.append(np.append(T_2.t, T_2.get_euler_angles()), 0))
        testlist.append(np.append(np.append(T_3.t, T_3.get_euler_angles()), 0))
        testlist.append(np.append(np.append(T_3.t, T_3.get_euler_angles()), 1))
        testlist.append(np.array([1]))
        testlist.append(np.append(np.append(T_4.t, T_4.get_euler_angles()), 1))
        testlist.append(np.append(np.append(T_5.t, T_5.get_euler_angles()), 1))
        testlist.append(np.append(np.append(T_5.t, T_5.get_euler_angles()), 0))
        testlist.append(np.append(np.append(T_6.t, T_6.get_euler_angles()), 0))
        print("Print out transform.")
        # print(np.append(T_1.t, T_1.get_euler_angles()))
        # print(np.append(T_2.t, T_2.get_euler_angles()))
        '''
        Example waypointlist:
        [np.array([x,y,z,euler1,2,3,gripper_boolean])]
        
        normal euler configuration: 0.66064978,
        -0.46117095,   2.09082787
        '''
        # print(testlist)
        first_position = np.array([146, 0, 45,  -np.pi, 0, 0, 1])
        offset = np.array([0,45,0,0,0,0,1])
        line_em_up_list = [first_position,first_position+offset,first_position+(2*offset),first_position+(3*offset)]
        self.tp.execute_plan(line_em_up_list, viz=False, look_ahead=4, controller = 'IK')
        self.current_state = "execute"
        self.next_state = "idle"
        self.rexarm.get_feedback()

    def manual(self):
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"
        self.rexarm.send_commands()
        self.rexarm.get_feedback()

    def idle(self):
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"
        self.rexarm.get_feedback()

    def estop(self):
        self.status_message = "EMERGENCY STOP - Check Rexarm and restart program"
        self.current_state = "estop"
        self.rexarm.disable_torque()
        self.rexarm.get_feedback()

    def calibrate(self):
        self.current_state = "calibrate"
        self.next_state = "idle"
        self.tp.go(max_speed=2.0)
        location_strings = ["lower left corner of board",
                            "upper left corner of board",
                            "upper right corner of board",
                            "lower right corner of board",
                            "center of shoulder motor"]
        i = 0
        for j in range(5):
            self.status_message = "Calibration - Click %s in RGB image" % location_strings[
                j]
            while (i <= j):
                self.rexarm.get_feedback()
                if(self.kinect.new_click == True):
                    self.kinect.rgb_click_points[
                        i] = self.kinect.last_click.copy()
                    i = i + 1
                    self.kinect.new_click = False

        i = 0
        for j in range(5):
            self.status_message = "Calibration - Click %s in depth image" % location_strings[
                j]
            while (i <= j):
                self.rexarm.get_feedback()
                if(self.kinect.new_click == True):
                    self.kinect.depth_click_points[
                        i] = self.kinect.last_click.copy()
                    i = i + 1
                    self.kinect.new_click = False
       
        """TODO Perform camera calibration here"""
        
        self.kinect.depth2rgb_affine = self.kinect.getAffineTransform(
            self.kinect.depth_click_points,
            self.kinect.rgb_click_points)
        camera_rgb_clicks = self.kinect.toCamearaCoord(self.kinect.rgb_click_points)
        print(camera_rgb_clicks)
        self.kinect.extrinsic = self.kinect.getAffineTransform(
            camera_rgb_clicks[:4, :2], self.kinect.fixed_coords)
        
        self.kinect.kinectCalibrated = True
        
        # save all clibration results
        np.save("./data/depth2rgb_affine.npy", self.kinect.depth2rgb_affine)
        np.save("./data/extrinsic.npy", self.kinect.extrinsic)
        
        self.status_message = "Calibration - Completed Calibration"
        time.sleep(1)
        """ Back to idle """
        self.next_state = "idle"
        time.sleep(0.2)
