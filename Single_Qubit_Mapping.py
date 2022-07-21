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
        DESCRIPTION. The default is CP2.
    num_steps : int, optional
        DESCRIPTION. The default is 1.
    num_sites : int, optional
        DESCRIPTION. The default is None.
    init_states : list, optional
        DESCRIPTION. The default is None.
    gate_list : list, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

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
            self.states = [np.array([0, 1]) for x in range(num_sites)]
        elif num_sites == None:
            self.states = [np.array(x) for x in init_states]

        if gate_list == None:
            self.gate_list = self.gen_gate_list()
        else:
            self.num_steps = len(gate_list)
            self.gate_list = gate_list
        self.gate_list.insert(0, np.identity(2))

        self.angles = []

        # self.final_unitary = self.gen_final_unitary()

    """
    Potentially slow but generates everything at once
    """

    def run(self):
        self.intermediate_states = self.gen_intermediate_states()
        self.angles = self.gen_angles()

    def gen_gate_list(self):
        vals = list(self.gate_set.values())
        return [random.choice(vals) for x in range(self.num_steps)]

    def gen_final_unitary(self):
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

    def gen_intermediate_states(self):
        return [[np.matmul(y, x) for y in self.gate_list] for x in self.states]

    def gen_final_state(self):
        return [np.matmul(self.final_unitary, x) for x in self.states]

    def gen_angles(self):
        return [
            [self.get_angle(state) for state in x] for x in self.intermediate_states
        ]

    def get_angle(self, state):
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


#%%
exp = Experiment(num_steps=5)
exp.run()

#%%
"""
Converts the two polar angles to 3D cartesean coordinates to be plotted
against the unit sphere

    angle_arr: list of polar angle pairs, usually called with arguement
    self.angles
"""


def U2_to_03Graph(angle_arr):

    x = np.array([get_x(angle_arr[0], angle_arr[1])])
    y = np.array([get_y(angle_arr[0], angle_arr[1])])
    z = np.array([get_z(angle_arr[0], angle_arr[1])])

    for k in range(len(angle_arr) // 2 - 1):

        x = np.append(x, np.array([get_x(angle_arr[2 * k + 2], angle_arr[2 * k + 3])]))
        y = np.append(y, np.array([get_y(angle_arr[2 * k + 2], angle_arr[2 * k + 3])]))
        z = np.append(z, np.array([get_z(angle_arr[2 * k + 2], angle_arr[2 * k + 3])]))

    plt.rcParams["figure.figsize"] = [20.00, 20.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    xin = [0]
    yin = [0]
    zin = [1]

    """
    First plot is of the bloch sphere, second is of all the generated data
    third is for labelling the state |0>
    """

    ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.3)
    ax.scatter(x, y, z, marker="o", color="black", alpha=0.9)
    ax.scatter(xin, yin, zin, marker="o", color="red")
    plt.show()


"""
Functions for getting cartesean coordinates from polar angles
"""


def get_x(theta, phi):

    return np.cos(phi) * np.sin(theta)


def get_y(theta, phi):

    return np.sin(phi) * np.sin(theta)


def get_z(theta, phi):

    return np.cos(theta)
