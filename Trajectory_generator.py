#!/usr/bin/env python

import math
from tkinter import W
import numpy as np
from scipy.optimize import minimize_scalar
import rospy
from geometry_msgs.msg import Vector3, Twist, Quaternion, Transform
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory

V_MAX = ( 2, 2, 4)
A_MAX = ( 1, 1, 2)
S_CAP = 0.05
ROSPY_RATE = 100
ALPHA = 0.01
ITER = 500
MAX_ITER = 2000


# Line 32 to 37 contain the equations of a0 to a5

FirstCall = True


def waypoint_publisher(poly_time_array):

    pointPub = rospy.Publisher("/red/position_hold/trajectory", MultiDOFJointTrajectoryPoint, queue_size = 1)
    total_time = 0
    poly_x_array = []
    poly_y_array = []
    poly_z_array = []
    rate = rospy.Rate(ROSPY_RATE)

    for x in poly_time_array:
        total_time += x[1]
        x[1] = total_time
        poly_x_array.append(x[0][0])
        poly_y_array.append(x[0][1])
        poly_z_array.append(x[0][2])

    num_paths = len(poly_time_array)
    dt = 1/ROSPY_RATE
    t0 = rospy.get_time()

    print("Goint to pubsli")

    print("poly_time_array : ",(poly_time_array))

    for i in range(0, math.ceil(total_time*ROSPY_RATE)):
        t = rospy.get_time() - t0
        t = t if t < total_time else total_time
        t_prev = 0
        for x in poly_time_array:
            if x[1]>t:
                t -= t_prev
                px = x[0][0]
                py = x[0][1]
                pz = x[0][2]
                vx = np.polyder(px)
                vy = np.polyder(py)
                vz = np.polyder(pz)
                px_ = px(t)
                py_ = py(t)
                pz_ = pz(t)
                vx_ = vx(t)
                vy_ = vy(t)
                vz_ = vz(t)

                rot = Quaternion(0,0,0,1)
                trans = Vector3(px_, py_, pz_)
                vel_lin = Vector3(vx_, vy_, vz_)
                vel_ang = Vector3(0,0,0)
                accel_lin = Vector3(0,0,0)
                accel_ang = Vector3(0,0,0)

                point = MultiDOFJointTrajectoryPoint(transforms=[Transform(translation = trans, rotation = rot)], velocities = [Twist(linear = vel_lin, angular = vel_ang)], accelerations = [Twist(linear = accel_lin, angular = accel_ang)])
                break
            t_prev = x[1]
        pointPub.publish(point)
        rate.sleep()


def WaypointCallback(msg):

    print("gotwaypoints.")

    waypoints = []

    for i in range(len(msg.points)):
        px = msg.points[i].transforms[0].translation.x
        py = msg.points[i].transforms[0].translation.y
        pz = msg.points[i].transforms[0].translation.z
        vx = msg.points[i].velocities[0].linear.x
        vy = msg.points[i].velocities[0].linear.y
        vz = msg.points[i].velocities[0].linear.z
        ax = msg.points[i].accelerations[0].linear.x
        ay = msg.points[i].accelerations[0].linear.y
        az = msg.points[i].accelerations[0].linear.z

        waypt = np.array([[px, vx, ax],
                          [py, vy, ay],
                          [pz, vz, az]])

        waypoints.append(waypt)

    poly_time_array = trajectory_with_time(waypoints)

    waypoint_publisher(poly_time_array)


def determine_coefficients(start, end, Spline_duration):
    # takes the start and end matrix in the form of
    # [[px, vx, ax],
    #  [py, vy, ay],
    #  [pz, vz, az]]
    # and the expected time duration

    # returns : polynomial array of [ax, ay, az]

    T = Spline_duration

    Ps = start[:, 0]
    Vs = start[:, 1]
    As = start[:, 2]
    Pe = end[:, 0]
    Ve = end[:, 1]
    Ae = end[:, 2]

    a0 = Ps
    a1 = Vs
    a2 = 1 / 2 * As
    a3 = 1 / (2 * (T ** 3)) * (20 * (Pe - Ps) - T * (8 * Ve + 12 * Vs) - T ** 2 * (3 * Ae - As))
    a4 = 1 / (2 * T ** 4) * (30 * (Ps - Pe) + T * (14 * Ve + 16 * Vs) + T ** 2 * (3 * As - 2 * Ae))
    a5 = 1 / (2 * T ** 5) * (12 * (Pe - Ps) - 6 * T * (Ve + Vs) - T ** 2 * (Ae - As))

    pol_mat = np.vstack((a5, a4, a3, a2, a1, a0))

    px = np.poly1d(pol_mat[:, 0])
    py = np.poly1d(pol_mat[:, 1])
    pz = np.poly1d(pol_mat[:, 2])

    pol_array = [px, py, pz]

    return pol_array


def determine_time(start, end, poly_mat=None, T_old=None):
    # takes input polynomial array and gives output of expected time
    # if no poly_mat and T_old are None then returns time from V_MAX

    if poly_mat is None and T_old is None:
        dist = math.sqrt(np.sum(np.square(end - start)))
        t = dist / V_MAX[2]
        return t, 1000

    time_list = []
    mul_list = []

    for axis in range(3):
        v = np.polyder(poly_mat[axis])
        a = np.polyder(poly_mat[axis], m=2)

        v_max_t = minimize_scalar(-v, bounds=[0, T_old], method='bounded', options={'maxiter':MAX_ITER})
        a_max_t = minimize_scalar(-a, bounds=[0, T_old], method='bounded', options={'maxiter':MAX_ITER})

        v_max = v(v_max_t.x)
        a_max = a(a_max_t.x)

        s = max( abs(v_max) / V_MAX[axis], math.sqrt( abs(a_max) / A_MAX[axis]))

        mul = s-1

        mul_list.append(mul)
        time_list.append(T_old * (1 + np.sign(mul) * ALPHA))

    T = max(time_list)
    mul = max(time_list)

    return T, mul


def spline_for_two_pts(start, end):
    """
    Order for the array to be given to determine coefficients is:
    [[x,  vx,  ax],
     [y, vy, ay],
     [z, vz, az]]
    for both start and end point

    and for Determine time we need an array of [px, py, pz]
    """
    # start = np.array([[0, 0, 0],
    #                   [0, 0, 0],
    #                   [-0.5, 0, 0]])
    # end = np.array([[10, 0, 0],
    #                 [0, 0, 0],
    #                 [0, 0, 0]])

    start_for_time = start[:, 0]
    end_for_time = end[:, 0]
    poly_mat = None
    t = None
    for x in range(ITER):
        t, mul = determine_time(start_for_time, end_for_time, poly_mat, t)
        if abs(mul) <= S_CAP:
            print(x, "iterations")
            break
        poly_mat = determine_coefficients(start, end, t)
    return poly_mat, t


def trajectory_with_time(waypoints):
    # The array of all the points in the trajectory from start to end
    path_time_list = []

    for x in range(len(waypoints)-1):
        print('determining coefficents for trajectory', x+1)
        poly, time = spline_for_two_pts(waypoints[x], waypoints[x+1])
        path_time_list.append([poly, time])

    return path_time_list


def sampling_rate(time):
    no_of_points = math.ceil(time * ROSPY_RATE)
    time_array = list(range(0, time, time/no_of_points))
    return time_array


def main():

    rospy.init_node("spline_traj_generator")

    rospy.Subscriber("/red/waypoints_to_spline", MultiDOFJointTrajectory, WaypointCallback)
    rospy.spin()


if __name__ == "__main__":
    main()
