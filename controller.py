import numpy as np


#Controller.py
#this file is the neural network that takes inputs and gives outputs

#genome: 8 inputs (joint angle and joint velocity for all 4 joints (2*4=8))
# 8 hidden neurons- close to input size and hopefully big enough to evolve good gait while not taking too long
#4 outputs: 1 torque for every joint

# w1+ b1 + w2 + b2 ----> (8*8)+ 8 + (8*4) + 4 = 108 genes
# w1=input*hidden | b1=bias per hidden neuron | w2=hidden * output | b2=output

#constants
HIDDEN=8        #keeps hidden layer same size regardless of size of snake

MOD_IN = 4
MOD_HIDDEN = 8
MOD_OUT = 1

#from before with n=5:
 #info from snake array : index 0- body orientation angle
    #                        index 1-4 - joint angles (joint 1 → joint 4)
    #                        index 5,6 - body x and y velocity
    #                        index 7 - body angular velocity
    #                        index 8-11 - joint angular velocities (joint 1 → joint 4)

def get_inputs(observation):
    # Works for any body length: joint angles start at index 1, joint velocities are the last values
    joints = (len(observation) - 4) // 2
    angles = observation[1:1 + joints]
    velocities = observation[-joints:]
    return np.concatenate([angles, velocities])

def genome_size(n_joints=4):
    # Centralized network: 2 inputs per joint, 1 output per joint
    n_in, n_out = 2 * n_joints, n_joints
    return n_in * HIDDEN + HIDDEN + HIDDEN * n_out + n_out   # 108 for N=5, 183 for N=8

def decode(genome, n_joints=4):
    # slice boundaries, each one = previous boundary + size of that piece
    n_in, n_out = 2 * n_joints, n_joints 
    end_w1 = n_in * HIDDEN              #64
    end_b1 = end_w1 + HIDDEN            #72
    end_w2 = end_b1 + HIDDEN * n_out    #104
    end_b2 = end_w2 + n_out             #108

     # W1: rows = hidden neurons, columns = inputs, so W1 @ x works
    w1 = genome[0:end_w1].reshape(HIDDEN, n_in)
    b1 = genome[end_w1:end_b1]

    # W2: rows = outputs, columns = hidden neurons, so W2 @ h works
    w2 = genome[end_b1:end_w2].reshape(n_out, HIDDEN)
    b2 = genome[end_w2:end_b2]
    return w1, b1, w2, b2



def act(w1, b1, w2, b2, inputs):        #takes inputs and weights and makes torques
    h=np.tanh((w1@inputs) + b1)

    torques= np.tanh((w2@h) + b2)

    return torques

# ----------------
# Modular Version
# ----------------

def get_inputs_modular(observation):

    joints = (len(observation)-4)//2 #this takes observation length, subtracts 4 (to get rid of indices 0,5,6,7) and divides by 2 to get correct number of joints

    angles = observation[1:joints+1]
    velocities = observation[len(observation)-joints:len(observation)]

    rows = []
    for j in range(joints):

        left=0.0
        right=0.0
        
        if j > 0:
            left = angles[j - 1]          #left is previous joints angle
        else: 0.0              # 0 if no left neighbor
        
        if j < joints - 1:
            right = angles[j + 1]       #next joints angle
        else: 0.0    # 0 if no right neighbor
        
        rows.append([angles[j], velocities[j], left, right])        

    return np.array(rows)   # shape (joints, 4): one row per joint

    #note the inputs for each joint controller are its own angle (input 1) and own velocity (input 2) and the angles of the joints to the left (input 3) and right (input 4)

def genome_size_modular():
    #total number of genes network needs

    return MOD_IN*MOD_HIDDEN + MOD_HIDDEN + MOD_HIDDEN*MOD_OUT + MOD_OUT   # = 49


def decode_modular(genome):
    # slice boundaries, each one = previous boundary + size of that piece
    end_w1 = MOD_IN * MOD_HIDDEN         #32   
    end_b1 = end_w1 + MOD_HIDDEN        #40
    end_w2 = end_b1 + MOD_HIDDEN * MOD_OUT   #48
    end_b2 = end_w2 + MOD_OUT    #49       

    # W1: rows = hidden neurons, columns = inputs, so W1 @ x works
    w1 = genome[0:end_w1].reshape(MOD_HIDDEN, MOD_IN)
    b1 = genome[end_w1:end_b1]
    # W2: rows = outputs, columns = hidden neurons, so W2 @ h works
    w2 = genome[end_b1:end_w2].reshape(MOD_OUT, MOD_HIDDEN)
    b2 = genome[end_w2:end_b2]

    return w1, b1, w2, b2


def act_modular(w1, b1, w2, b2, inputs):
    # inputs: (joints, 4) array, one row per joint
    # The same small network runs on every joint's row
    torques = []
    for x in inputs:
        h = np.tanh(w1 @ x + b1)        # hidden layer: (8, 4) @ (4,) + (8,) -> (8,)
        t = np.tanh(w2 @ h + b2)        # output: (1, 8) @ (8,) + (1,) -> (1,)
        torques.append(t[0])            # grab the single number out of the 1-element array
    return np.array(torques)            # shape (joints,): one torque per joint


    