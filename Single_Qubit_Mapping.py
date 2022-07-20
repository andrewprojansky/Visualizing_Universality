'''
Code to visualize the affect of random unitaries composed of fundamental
gate sets on different initial states, with the goal of understanding
universality, and magic states/gates.

Code Written by Andrew Projansky
Project Start Date: 7/18/2022
Most recent update: 7/19/2022
'''

import numpy as np
import matplotlib.pyplot as plt
import random

'''
Defines all gates used
'''

H = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
S = np.array([[1,0],[0,1j]])
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])

'''
Defines the dictionaries of different gate sets: currently,
just the hadamard, the one dimensional clifford group,
and the clifford plus the pauli matrices
'''

C0 = {1:H}
C1 = {1:H,2:S}
CP1 = {1:H,2:S,3:Z,4:X,5:Y}

'''
Defines possible initial states

note: still need to add in an arbitrary state
'''

th = np.arccos(1/np.sqrt(3))/2
IS1_1 = np.array([1.0,0.0])
IS2_1 = np.array([np.cos(th),np.exp(np.pi/4 * 1j)*np.sin(th)])
#IS3_1 = 'random'

'''
Circuit_Run: class for defining a trial

    gate_num: number of gates to be randomly chosen
    gate_set: set of gates to be randomly chosen from
    state: inital state
    runs: number of data points for given number of gates and inital state
'''

class Circuit_Run:

    def __init__(self, gate_num, gate_set, state, runs):

        self.gate_num = gate_num
        self.gate_set = gate_set
        self.state = state
        self.gate_choices = len(gate_set)
        self.runs = runs
        self.final_unitary = np.zeros((2,2))
        self.final_state = np.zeros(2)
        self.angles = np.zeros((runs,2))

    '''
    total_unitary: performs matrix mutiplication to compose total unitary
    from multiplcation of randomly selected gates
    '''

    def total_unitary(self):

            gate_step_i = self.gate_set[random.randint(1,self.gate_choices)]
            if self.gate_num > 1:
                for i in range(self.gate_num -1):
                    gate_step_j = self.gate_set[random.randint(1,self.gate_choices)]
                    gate_step_i = np.matmul(gate_step_j,gate_step_i)
            self.final_unitary = gate_step_i

    '''
    u_on_state: applies total unitary onto initial state
    '''

    def u_on_state(self):

            self.final_state = np.matmul(self.final_unitary, self.state)

    '''
    from the final state, taking the state amplitude and converting them into
    the angles which define the arbitrary single qubit state
    '''

    def get_angles(self):

            first_mag = np.sqrt(np.real(self.final_state[0])**2 + np.imag(self.final_state[0])**2)
            second_mag = np.sqrt(np.real(self.final_state[1])**2 + np.imag(self.final_state[1])**2)
            first_phi = np.arctan2(np.imag(self.final_state[0]),np.real(self.final_state[0]))
            second_phi = np.arctan2(np.imag(self.final_state[1]),np.real(self.final_state[1]))
            phi = second_phi - first_phi
            theta = 2*np.arccos(np.real(self.final_state[0] * np.exp(-1j*first_phi)))
            self.angles = np.append(self.angles,[[theta,phi]])

    '''
    For the number of trials, repeats the circuit constructor
    '''

    def random_circuits(self):

            for j in range(self.runs):

                self.total_unitary()
                self.u_on_state()
                self.get_angles()
                if j == 0:
                    self.angles = self.angles[2*self.runs:]
'''
Converts the two polar angles to 3D cartesean coordinates to be plotted
against the unit sphere

    angle_arr: list of polar angle pairs, usually called with arguement
    self.angles
'''

def U2_to_03Graph(angle_arr):

    x = np.array([get_x(angle_arr[0],angle_arr[1])])
    y = np.array([get_y(angle_arr[0],angle_arr[1])])
    z = np.array([get_z(angle_arr[0],angle_arr[1])])

    for k in range(len(angle_arr)//2 - 1):

        x = np.append(x, np.array([get_x(angle_arr[2*k+2],angle_arr[2*k+3])]))
        y = np.append(y, np.array([get_y(angle_arr[2*k+2],angle_arr[2*k+3])]))
        z = np.append(z, np.array([get_z(angle_arr[2*k+2],angle_arr[2*k+3])]))

    plt.rcParams["figure.figsize"] = [20.00, 20.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color='lightgrey', alpha=0.3)
    ax.scatter(x, y, z, marker='o', color='black')
    plt.show()


'''
Functions for getting cartesean coordinates from polar angles
'''

def get_x(theta, phi):

    return np.cos(phi) * np.sin(theta)

def get_y(theta, phi):

      return np.sin(phi) * np.sin(theta)

def get_z(theta, phi):

    return np.cos(theta)

'''






'''

def main():

    pass
    #fill in with class instance and whatever else you want

main()
