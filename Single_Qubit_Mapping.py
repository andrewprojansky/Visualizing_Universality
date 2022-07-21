"""
Code to visualize the affect of random unitaries composed of fundamental
gate sets on different initial states, with the goal of understanding
universality, and magic states/gates.

Code Written by Andrew Projansky
hahahahaha -Joe
Project Start Date: 7/18/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import random

"""
Defines all gates used
"""

H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
S = np.array([[1, 0], [0, 1j]])
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
T = np.array([[np.exp(-1j * np.pi / 8), 0], [0, np.exp(1j * np.pi / 8)]])
univ_rot = np.matmul(T, np.matmul(H, np.matmul(T, H)))

"""
Defines the dictionaries of different gate sets: currently,
the hadamard, the one dimensional clifford group, the hadamard and t gate,
the one dimensional clifford plus T, and the clifford plus the pauli matrices

currently for testing we have a gate set that is just a single matrix,
THTH - the irrational rotation
"""

C0 = {1: H}
C1 = {1: H, 2: S}
CP2 = {1: H, 2: T}
CP3 = {1: H, 2: S, 3: T}
CP1 = {1: H, 2: S, 3: Z, 4: X, 5: Y}
test = {1: univ_rot}

"""
Defines possible initial states

note: still need to add in an arbitrary state
"""

th = np.arccos(1 / np.sqrt(3)) / 2
IS1_1 = np.array([1.0, 0.0])
IS2_1 = np.array([np.cos(th), np.exp(np.pi / 4 * 1j) * np.sin(th)])
# IS3_1 = 'random'

"""
Circuit_Run: class for defining a trial

    gate_num: number of gates to be randomly chosen
    gate_set: set of gates to be randomly chosen from
    state: inital state
    runs: number of data points for given number of gates and inital state
"""
#%%
"""
possible addition if we want fancy states
class state:
    def __init__(self, init_state:numpy.ndarray =np.array([0,1])):
        self.state = init_state
"""


class Experiment:
    """
    Parameters
    ----------
    gate_set : dict, optional
        set of gates to use. The default is CP2.
    num_steps : int, optional
        total number of gates. The default is 1.
    num_sites : int, optional
        number of independent state positions. The default is None.
    init_states : list, optional
        initial position. The default is None.
    gate_list : list, optional
        List of gates to be applied, if not suppplied will generate random
        gates from gate set. The default is None.

    """

    def __init__(
        self,
        gate_set: dict = CP2,
        num_steps: int = 1,
        num_sites: int = None,
        init_states: list = None,
        gate_list: list = None,
    ):

        self.num_steps = num_steps
        self.gate_set = gate_set

        if num_sites == None and init_states == None:
            self.states = [np.array([0, 1])]
        elif init_states == None:
            """TODO make random init positions"""
            self.states = [np.array([0, 1]) for x in range(num_sites)]
        elif num_sites == None:
            self.states = [np.array(x) for x in init_states]

        if gate_list == None:
            self.gate_list = self.__gen_gate_list()
        else:
            self.num_steps = len(gate_list)
            self.gate_list = gate_list
        # self.gate_list.insert(0, np.identity(2))

        self.angles = []

        # self.final_unitary = self.__gen_final_unitary()

    def __gen_gate_list(self):
        vals = list(self.gate_set.values())
        return [random.choice(vals) for x in range(self.num_steps)]

    ################# run functions ##################################
    def run(self):
        self.intermediate_states = self.__gen_intermediate_states()
        self.angles = self.__gen_angles()

    def __gen_final_unitary(self):
        """
        performs matrix mutiplication to compose total unitary
        from multiplcation of randomly selected gates
        varibials-
        """
        gate_step_i = self.gate_list[0]
        for i in range(1, num_steps):
            gate_step_j = self.gate_list(i)
            gate_step_i = np.matmul(gate_step_j, gate_step_i)
        return gate_step_i

    def __gen_intermediate_states(self):
        inter_gate_list = self.gen_intermediate_unitaries()
        return [[np.matmul(y, x) for y in inter_gate_list] for x in self.states]

    def gen_intermediate_unitaries(self):
        unitary_arr = [np.identity(2)]
        for i in range(1, self.num_steps):
            unitary_arr.append(np.matmul(unitary_arr[i - 1], self.gate_list[i]))
        return unitary_arr

    # def __gen_final_state(self):
    #     return [np.matmul(self.final_unitary, x) for x in self.states]

    def __gen_angles(self):
        return [
            [self.__get_angle(state) for state in x] for x in self.intermediate_states
        ]

    def __get_angle(self, state):
        """
        get an angle for a single state
        """
        first_mag = np.sqrt(np.real(state[0]) ** 2 + np.imag(state[0]) ** 2)
        second_mag = np.sqrt(np.real(state[1]) ** 2 + np.imag(state[1]) ** 2)
        first_phi = np.arctan2(np.imag(state[0]), np.real(state[0]))
        second_phi = np.arctan2(np.imag(state[1]), np.real(state[1]))
        phi = second_phi - first_phi
        theta = 2 * np.arccos(np.real(state[0] * np.exp(-1j * first_phi)))

        return [theta, phi]

    ##############plot functions ########################
    def plot(self):
        states_xyz = [self.__gen_xyz_points(x) for x in self.angles]
        print(np.shape(states_xyz))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)

        ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.3)
        for x in states_xyz:
            ax.scatter(x[0], x[1], x[2], marker="o", alpha=0.9)
        # ax.scatter(xin, yin, zin, marker="o", color="red")
        plt.show()

    def __gen_xyz_points(self, angle_arr):
        x = np.array([self.__get_x(angle_arr[0], angle_arr[1])])
        y = np.array([self.__get_y(angle_arr[0], angle_arr[1])])
        z = np.array([self.__get_z(angle_arr[0], angle_arr[1])])

        for k in range(len(angle_arr) // 2 - 1):
            x = np.append(
                x, np.array([self.__get_x(angle_arr[2 * k + 2], angle_arr[2 * k + 3])])
            )
            y = np.append(
                y, np.array([self.__get_y(angle_arr[2 * k + 2], angle_arr[2 * k + 3])])
            )
            z = np.append(
                z, np.array([self.__get_z(angle_arr[2 * k + 2], angle_arr[2 * k + 3])])
            )

        return x, y, z

    def __get_x(self, theta, phi):
        return np.cos(phi) * np.sin(theta)

    def __get_y(self, theta, phi):
        return np.sin(phi) * np.sin(theta)

    def __get_z(self, theta, phi):
        return np.cos(theta)


#%%
n = np.sqrt(1 / 2)
init_state = [[1, 0], [0, 1], [n, n]]
exp = Experiment(init_states=init_state, num_steps=200)
exp.run()
exp.plot()
