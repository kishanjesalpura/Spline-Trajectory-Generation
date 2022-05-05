import math
import numpy as np
from scipy.optimize import minimize_scalar

V_MAX = 4.5
A_MAX = 5
S_CAP = 0.1
ROSPY_RATE = 20

# Line 32 to 37 contain the equations of a0 to a5

FirstCall = True


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
        t = dist / V_MAX
        return t, 1000

    v_x = np.polyder(poly_mat[0])
    v_y = np.polyder(poly_mat[1])
    v_z = np.polyder(poly_mat[2])
    a_x = np.polyder(poly_mat[0], m=2)
    a_y = np.polyder(poly_mat[1], m=2)
    a_z = np.polyder(poly_mat[2], m=2)

    v_net_sq = (v_x * v_x + v_y * v_y + v_z * v_z)
    a_net_sq = (a_x * a_x + a_y * a_y + a_z * a_z)

    v_max_t = minimize_scalar(-v_net_sq, bounds=[0, T_old], method='bounded')
    a_max_t = minimize_scalar(-a_net_sq, bounds=[0, T_old], method='bounded')

    v_max = v_net_sq(v_max_t.x)
    a_max = a_net_sq(a_max_t.x)

    v_max = math.sqrt(v_max)
    a_max = math.sqrt(a_max)

    print(v_max, "vamx")

    s = max(v_max / V_MAX, math.sqrt(a_max / A_MAX))
    mul = s - 1

    T = T_old * (1 + np.sign(mul) * 0.01)

    return T, mul


def spline_for_two_pts(start, end):
    """
    Order for the array to be given to determine coefficents is:
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
    for x in range(1000):
        t, mul = determine_time(start_for_time, end_for_time, poly_mat, t)
        if abs(mul) <= S_CAP:
            break
        poly_mat = determine_coefficients(start, end, t)
    return poly_mat, t


def trajectory_with_time():
    # The array of all the points in the trajectory from start to end
    array = []
    path_time_list = []

    for x in range(len(array)-1):
        poly, time = spline_for_two_pts(array[x], array[x+1])
        path_time_list.append = [poly, time]

    return path_time_list


def sampling_rate(time):
    no_of_points = math.ceil(time * ROSPY_RATE)
    time_array = list(range(0, time, time/no_of_points))
    return time_array


if __name__ == "__main__":
