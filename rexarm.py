#rexarm.py
import numpy as np
from kinematics import *
import time
import serial
import kinematics

""" 
TODO:

Implement the missing functions
add anything you see fit

"""

""" Radians to/from  Degrees conversions """
D2R = 3.141592/180.0
R2D = 180.0/3.141592

class Rexarm():
    def __init__(self, joints, gripper):
        #self.gripper_ser = serial.Serial('/dev/ttyACM0', 9600)
        
        self.joints = joints
        self.gripper = gripper
        self.gripper_open_pos = np.deg2rad(-60.0)
        self.gripper_closed_pos = np.deg2rad(30.0)
        self.gripper_state = True
        self.estop = False
        """TODO: Find the physical angle limits of the Rexarm. Remember to keep track of this if you include more motors"""
        self.angle_limits = np.array([
                            [-180, 179.99],
                            [-90, 90.0],
                            [-116.0, 107.0],
                            [-180, 179.99],
                            [-135.0, 45],
                            [-135, 45]], dtype=np.float)*D2R

        """ Commanded Values """
        self.num_joints = len(joints)
        print(self.num_joints)
        self.position = [0.0] * self.num_joints     # degrees
        self.speed = [1.0] * self.num_joints        # 0 to 1
        self.max_torque = [1.0] * self.num_joints   # 0 to 1

        """ Feedback Values """
        self.joint_angles_fb = [0.0] * self.num_joints # degrees
        self.speed_fb = [0.0] * self.num_joints        # 0 to 1   
        self.load_fb = [0.0] * self.num_joints         # -1 to 1  
        self.temp_fb = [0.0] * self.num_joints         # Celsius
        self.move_fb = [0] *  self.num_joints

    def initialize(self):
        for joint in self.joints:
            joint.enable_torque()
            joint.set_position(0.0)
            joint.set_torque_limit(0.5)
            joint.set_speed(0.25)
        if(self.gripper != 0):
            self.gripper.set_torque_limit(1.0)
            self.gripper.set_speed(0.8)
            self.close_gripper()
        self.joints[3].set_compliance(4,64)
        self.joints[4].set_compliance(4,64)

    def open_gripper(self):
        """ TODO """
        self.gripper_state = False
        #self.gripper_ser.write(b'0')
        print('rexarm: Opened gripper')

    def close_gripper(self):
        """ TODO """
        self.gripper_state = True
        #self.gripper_ser.write(b'1')
        print('rexarm: Closed gripper')

    def toggle_gripper(self):
        """ TODO """
        pass

    def set_positions(self, joint_angles, update_now = True):
        joint_angles = np.array(joint_angles)
        self.clamp(joint_angles)
        T_print = kinematics.FK_dh(joint_angles, 6)
        print('current pose: ', np.append(T_print.t, T_print.get_euler_angles()))
        try:
            for i,joint in enumerate(self.joints):
                self.position[i] = joint_angles[i]
                if(update_now):
                    joint.set_position(joint_angles[i])
        except:
            return 1
    
    def set_speeds_normalized_global(self, speed, update_now = True):
        for i,joint in enumerate(self.joints):
            self.speed[i] = speed
            if(update_now):
                joint.set_speed(speed)

    def set_speeds_normalized(self, speeds, update_now = True):
        for i,joint in enumerate(self.joints):
            self.speed[i] = speeds[i]
            if(update_now):
                joint.set_speed(speeds[i])

    def set_speeds(self, speeds, update_now = True):
        for i,joint in enumerate(self.joints):
            self.speed[i] = speeds[i]
            speed_msg = abs(speeds[i]/joint.max_speed) #?/(rad/s)
            if (speed_msg < 3.0/1023.0):
                speed_msg = 3.0/1023.0
            if(update_now):
                joint.set_speed(speed_msg)
    
    def set_torque_limits(self, torques, update_now = True):
        for i,joint in enumerate(self.joints):
            self.max_torque[i] = torques[i]
            if(update_now):
                joint.set_torque_limit(torques[i])

    def send_commands(self):
        self.set_positions(self.position)
        self.set_speeds_normalized(self.speed)
        self.set_torque_limits(self.max_torque)

    def enable_torque(self):
        for joint in self.joints:
            joint.enable_torque()

    def disable_torque(self):
        for joint in self.joints:
            joint.disable_torque()

    def get_positions(self):
        for i,joint in enumerate(self.joints):
            self.joint_angles_fb[i] = joint.get_position()
        return self.joint_angles_fb

    def get_speeds(self):
        for i,joint in enumerate(self.joints):
            self.speed_fb[i] = joint.get_speed()
        return self.speed_fb

    def get_loads(self):
        for i,joint in enumerate(self.joints):
            self.load_fb[i] = joint.get_load()
        return self.load_fb

    def get_temps(self):
        for i,joint in enumerate(self.joints):
            self.temp_fb[i] = joint.get_temp()
        return self.temp_fb

    def get_moving_status(self):
        for i,joint in enumerate(self.joints):
            self.move_fb[i] = joint.is_moving()
        return self.move_fb

    def get_feedback(self):
        self.get_positions()
        self.get_speeds()
        self.get_loads()
        self.get_temps()
        self.get_moving_status()

    def pause(self, secs):
        time_start = time.time()
        while((time.time()-time_start) < secs):
            self.get_feedback()
            time.sleep(0.05)
            if(self.estop == True):
                break

    def clamp(self, joint_angles):
        """TODO"""
        np.clip(joint_angles, self.angle_limits.T[0], self.angle_limits.T[1], out=joint_angles)
    def get_wrist_pose(self):
        """TODO"""
        return [0,0,0,0,0,0]
