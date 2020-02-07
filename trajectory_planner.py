import numpy as np
import time

import matplotlib.pyplot as plt

import kinematics
from kinematics_convert import *

COLORS = ['tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:brown']

"""
TODO: build a trajectory generator and waypoint planner 
        so it allows your state machine to iterate through
        the plan at the desired command update rate
"""


class TrajectoryPlanner():

    def __init__(self, rexarm):
        self.idle = True
        self.rexarm = rexarm
        self.num_joints = rexarm.num_joints
        self.initial_wp = np.array([0.0] * self.num_joints)
        self.final_wp = np.array([0.0] * self.num_joints)
        self.dt = 0.02  # command rate
        self.history = np.copy(self.rexarm.joint_angles_fb)
        self.world_history = kinematics.FK_dh(self.history[0], 6).t
        self.grip = 0
        self.traj_obj = [np.zeros(6),np.zeros(6)]

    def set_initial_wp(self):
        self.initial_wp = np.copy(self.rexarm.joint_angles_fb)

    def set_final_wp(self, waypoint):
        self.final_wp = np.array(waypoint)

    def go(self, max_speed=2.5, lookahead=3):
        self.idle = False
        print('trajectory_planner: Going')
        T = self.calc_time_from_waypoints(
            self.initial_wp, self.final_wp, max_speed)

        #spline = self.generate_cubic_spline(self.initial_wp, self.final_wp, T)
        spline = self.generate_linear_spline(self.initial_wp, self.final_wp, T)
        pointlist = spline[0]
        v_list = spline[1]
        print('pointlist:',pointlist, len(pointlist))
        print('vlist:',v_list, len(v_list))

        for point_i in range(len(pointlist)):
            # self.rexarm.set_speeds_normalized_global(max_speed)
            '''
            mhe = 0
            for i in range(mhe_points):
                if point_i + i >= len(pointlist):
                    mhe += pointlist[point_i]
                else:
                    mhe += pointlist[point_i + i]

            mhe_divided = mhe / mhe_points
            self.rexarm.set_positions(mhe_divided)
            '''

            speed_i = v_list[point_i]*.35*np.array([1,1,1,1,1.5,1])
            #speed_i *= .04  # magic number
            self.rexarm.set_speeds(speed_i)
            if point_i + lookahead <= len(pointlist) - 1:
                fail = self.rexarm.set_positions(pointlist[point_i + lookahead])
                if fail == 1:
                    self.plot_traj()
            elif point_i == 0:
                self.rexarm.set_positions(pointlist[-1])

            self.record_joints()
            self.rexarm.pause(self.dt)
        #time.sleep(.25)
        return spline

    def stop(self):
        self.idle = True

    def pause(self, secs):
        print('trajectory_planner: Pausing for {} secs'.format(secs))
        self.rexarm.pause(secs)

    def calc_time_from_waypoints(self, initial_wp, final_wp, max_speed):
        max_delta_angle = max(abs(initial_wp - final_wp))
        print(max_delta_angle)
        return (3 * max_delta_angle) / (2 * max_speed)

    def record_joints(self):
        self.history = np.vstack(
            [self.history, np.copy(self.rexarm.joint_angles_fb)])
        self.world_history = np.vstack(
            [self.world_history, kinematics.FK_dh(np.copy(self.rexarm.joint_angles_fb), 6).t])

    def generate_linear_spline(self, initial_wp, final_wp, T):
        

        num_points = int(T / self.dt)

        timescaled_cubic = [np.linspace(0, 1, num_points)]
        timescaled_cubic.append([np.copy(timescaled_cubic[0])])
        timescaled_cubic.append([0*timescaled_cubic[0] + 1])
        timescaled_cubic[0] *= T

        delta_theta = final_wp - initial_wp

        list_of_interpolated_points = (delta_theta[
            0] * np.array(timescaled_cubic[1])) + initial_wp[0]
        list_of_derivatives = (delta_theta[
            0] * np.array(timescaled_cubic[2]))

        for i in range(1, self.num_joints):
            list_of_interpolated_points = np.vstack([
                list_of_interpolated_points, ((delta_theta[i] * np.array(timescaled_cubic[1])) + initial_wp[i])])
            list_of_derivatives = np.vstack([
                list_of_derivatives, ((delta_theta[i] * np.array(timescaled_cubic[2])))])

        return [np.transpose(list_of_interpolated_points), np.transpose(list_of_derivatives)]
    
    def generate_cubic_spline(self, initial_wp, final_wp, T):
        '''
        joint_parameters format:
        [theta1, theta2, theta3, theta4, theta5]
        '''

        num_points = int(T / self.dt)

        timescaled_cubic = [np.linspace(0, 1, num_points)]
        timescaled_cubic.append(
            [(3 * timescaled_cubic[0]**2) - (2 * timescaled_cubic[0]**3)])
        timescaled_cubic.append(
            [(6 * timescaled_cubic[0]) - (6 * timescaled_cubic[0]**2)])
        timescaled_cubic[0] *= T

        delta_theta = final_wp - initial_wp

        list_of_interpolated_points = (delta_theta[
            0] * np.array(timescaled_cubic[1])) + initial_wp[0]
        list_of_derivatives = (delta_theta[
            0] * np.array(timescaled_cubic[2]))

        for i in range(1, self.num_joints):
            list_of_interpolated_points = np.vstack([
                list_of_interpolated_points, ((delta_theta[i] * np.array(timescaled_cubic[1])) + initial_wp[i])])
            list_of_derivatives = np.vstack([
                list_of_derivatives, ((delta_theta[i] * np.array(timescaled_cubic[2])))])

        return [np.transpose(list_of_interpolated_points), np.transpose(list_of_derivatives)]

    def execute_plan(self, plan, look_ahead=0, viz=False, controller='indep_joint'):
        '''
        Iterate through waypoints specified by:
        plan=[np.array([j1,j2,j3,j4,j5]),...]
        viz allows for plotting of the trajectories using matplotlib
        '''
        if controller == 'IK':

            for waypoint in plan:
                if np.shape(waypoint) == (1,):
                    # Pause type waypoint
                    self.set_initial_wp()
                    self.pause(waypoint[0])
                    num_points = int(.25*waypoint[0] / self.dt)
                    
                    curr_traj_obj = [np.vstack([self.initial_wp] * num_points), np.vstack([np.zeros(6)] * num_points)]
                    curr_world = np.vstack([kinematics.FK_dh(self.initial_wp, 6).t] * num_points)
                    self.history = np.vstack([self.history, curr_traj_obj[0]])
                    self.world_history = np.vstack([self.world_history, curr_world])
                else:
                    # Movement type waypoint
                    joint_waypoint = kinematics.IK(get_T_from_euler_pos(waypoint[3:-1],waypoint[:3]))
                    if waypoint[-1] == 1:
                        self.rexarm.close_gripper()
                        self.grip = 1
                    if waypoint[-1] == 0:
                        self.rexarm.open_gripper()
                        self.grip = 0

                    self.set_initial_wp()
                    self.set_final_wp(joint_waypoint)

                    num_points = 0#int(.1/self.dt)
                    #curr_pause_obj = [np.vstack([self.final_wp] * num_points), np.vstack([np.zeros(6)] * num_points)]
                    curr_traj_obj = self.go(max_speed=2.5, lookahead=look_ahead) #high speed = 4
                    #curr_traj_obj[0] = np.vstack([curr_traj_obj[0],curr_pause_obj[0]])
                    #curr_traj_obj[1] = np.vstack([curr_traj_obj[1],curr_pause_obj[1]])

                self.traj_obj[0] = np.vstack([self.traj_obj[0], curr_traj_obj[0]])
                self.traj_obj[1] = np.vstack([self.traj_obj[1], curr_traj_obj[1]])
            
            self.stop()

        if controller == 'indep_joint':
            print('trajectory_planner: using independent joint control')

            traj_list = 0
            v_list = 0

            for waypoint in plan:
                if np.shape(waypoint) == (1,):
                    # Pause type waypoint
                    self.set_initial_wp()
                    self.pause(waypoint[0])
                    curr_traj = self.generate_cubic_spline(
                        self.initial_wp, self.initial_wp, waypoint[0])[0]
                else:
                    # Movement type waypoint
                    self.set_initial_wp()
                    self.set_final_wp(waypoint)

                    curr_traj_obj = self.go(max_speed=1, lookahead=look_ahead)
                    curr_traj = curr_traj_obj[0]
                    curr_v = curr_traj_obj[1]

                # Add waypoint to list for plotting

                if type(traj_list) == int:
                    traj_list = curr_traj
                    v_list = curr_v
                else:
                    traj_list = np.vstack([traj_list, curr_traj])
                    v_list = np.vstack([v_list, curr_v])

            self.stop()

            if viz:
                fig = plt.figure()
                ax1 = plt.subplot(121)
                for i in range(5):
                    ax1.plot(np.transpose(traj_list)[
                             i], label='Joint {} command'.format(i + 1), color=COLORS[i], linestyle='--')
                    ax1.plot(np.transpose(self.history)[
                             i], label='Joint {} trajectory'.format(i + 1), color=COLORS[i], linestyle='-')
                ax1.legend(loc='best')
                # Velocity plotting
                ax2 = plt.subplot(122)
                for i in range(5):
                    ax2.plot(np.transpose(v_list)[
                             i], label='Joint {} command'.format(i + 1), color=COLORS[i], linestyle='--')
                    #ax2.plot(np.transpose(self.history)[i], label='Joint {} trajectory'.format(i + 1), color=COLORS[i], linestyle='-')
                ax2.legend(loc='best')
                
                plt.tight_layout()
                plt.show()
                
    def plot_traj(self):
        # 2-element list of arrays, nx6
        # array, nx6
        np.savetxt('high_rough_coord.csv', self.world_history, delimiter=',')
        coord_lookup = ['X', 'Y', 'Z']
        
        fig = plt.figure()
        ax1 = plt.subplot(121)
        for i in range(3):
            ax1.plot(np.transpose(self.world_history)[
                        i], label='{} position, world'.format(coord_lookup[i]), color=COLORS[i], linestyle='-')
        ax1.legend(loc='best')
        # for i in range(6):
        #     ax1.plot(np.transpose(self.traj_obj[0])[
        #                 i], label='Joint {} command'.format(i + 1), color=COLORS[i], linestyle='--')
        #     ax1.plot(np.transpose(self.history)[
        #                 i], label='Joint {} encoder reading'.format(i + 1), color=COLORS[i], linestyle='-')
        # ax1.legend(loc='best')
        # # Velocity plotting
        # ax2 = plt.subplot(122)
        # for i in range(6):
        #     ax2.plot(np.transpose(self.traj_obj[1])[
        #                 i], label='Joint {} command'.format(i + 1), color=COLORS[i], linestyle='--')
        #     #ax2.plot(np.transpose(self.history)[i], label='Joint {} trajectory'.format(i + 1), color=COLORS[i], linestyle='-')
        # ax2.legend(loc='best')
        
        plt.tight_layout()
        plt.show()

'''
import rexarm
R_example = rexarm.Rexarm([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

T_example = TrajectoryPlanner(R_example)
wp0 = np.random.random(5)
wp1 = np.random.random(5)
wp2 = np.random.random(5)

traj1 = T_example.generate_cubic_spline(wp0, wp1, 1)
traj2 = T_example.generate_cubic_spline(wp1, wp2, 1)

traj = np.vstack([traj1, traj2])

fig = plt.figure()

ax1 = plt.subplot(111)

for i in range(0, 5):
    ax1.plot(np.transpose(traj)[i], label=i)
ax1.legend(loc='best')

plt.tight_layout()

plt.show()
'''
