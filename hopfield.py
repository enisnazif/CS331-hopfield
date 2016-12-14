import numpy as np
import random
import itertools
from copy import deepcopy

### Inhibition_constant can be 0.5 or 1
INHIBITION = 1

### Number of units in the network
N = 9

### Table of initial states chosen at random
init = [[1, 1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1],
[1, 1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1, 0, 1, 1],
[1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1, 1, 1],
[0, 0, 1, 0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0],
[0, 1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0],
[0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 1, 0, 1, 1],
[1, 0, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0],
[1, 0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0, 0, 1],
[0, 0, 1, 0, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1],
[0, 0, 0, 0, 1, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0]];


# Declare an empty array
weight_matrix = np.empty([9, 9], dtype=float)
# Fill the matrix with negative inhibition constants
weight_matrix.fill(-INHIBITION)
# By the Hebb rule, the state we are trying to store is:
# (0 1 0 1 1 1 0 1 0 ) so set the weights accordingly
weight_matrix[np.ix_([1],[4,7])] = 1
weight_matrix[np.ix_([4],[1,3,5,7])] = 1
weight_matrix[np.ix_([7],[1,4])] = 1
weight_matrix[np.ix_([3],[4,5])] = 1
weight_matrix[np.ix_([5],[4,3])] = 1
# Set all the elements on the leading diagonal to 0, since w_ii = 0 for all i
np.fill_diagonal(weight_matrix, 0)

print 'Weight matrix is:'
print weight_matrix
print ''

### Network energy function that calculates the total energy in the network for a given state
def energy(state):
    network_energy = 0
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            network_energy = network_energy + weight_matrix[i,j]*state[0,i]*state[0,j]

    return -1/2*network_energy

### Peforms a synchronous update of the neuron weights
def sync_update(weight_matrix, state, iterations):
    for n in range(iterations):
        for i in range(9):
            previous_state = state
            update = weight_matrix[i]*np.transpose(previous_state)
            if (update > 0):
                state[0,i] = 1
            else:
                state[0,i] = 0

    return state

### Performs an asynchronous update of the neuron weights in sequential order
def async_update(weight_matrix, state, iterations):
    for n in range(iterations):
        previous_state = state
        unit = random.randrange(0, N)
        update = weight_matrix[unit]*np.transpose(previous_state)
        if (update > 0):
            state[0,unit] = 1
        else:
            state[0,unit] = 0

    return state

### Runs the Hopfield network
def hopfield(weight_matrix, state, method, cycles):
    return method(weight_matrix, state, cycles)

### Function that determines whether or not a given state is a fixed point of the network
def is_fixed_point(state):
    return (np.sum(np.matrix(state) == hopfield(weight_matrix, np.matrix(state), sync_update, 10)) == 9) == True

### Q1
# Get each of the 2^9 binary states to test
binary_states = list(itertools.product([0, 1], repeat=9))
fixed_points = []
# For each binary state, test if it is a fixed point of the network
print 'Fixed Points of the Network: '
for s in binary_states:
    if is_fixed_point(s):
        fixed_points.append(s)
        print s

print ' '
## Running with inhibition constant of 0.5 gives four fixed points:
# (0, 0, 0, 0, 0, 0, 0, 0, 0)
# (0, 0, 0, 1, 1, 1, 0, 0, 0)
# (0, 1, 0, 0, 1, 0, 0, 1, 0)
# (0, 1, 0, 1, 1, 1, 0, 1, 0)

## Running with inhibition constant of 1 gives
# (0, 0, 0, 0, 0, 0, 0, 0, 0)
# (0, 0, 0, 1, 1, 1, 0, 0, 0)
# (0, 1, 0, 0, 1, 0, 0, 1, 0)

### Q2
# Function that gets the 9 states that are at most a Hamming distance of 1 away from 'state'
def get_neighbour_states(state):
    neighbours = []
    for i in range(9):
        state_copy = deepcopy(state)
        state_copy[0,i] = np.abs(state_copy[0,i]-1)
        neighbours.append(state_copy)

    return(neighbours)

# For each fixed point, get the 9 states that are at most a Hamming distance of 1 away
for s in fixed_points:
    neighbours = get_neighbour_states(np.matrix(s))
    print 'Energy of Fixed Point: ' + str(np.matrix(s)) + ' = ' + str(energy(np.matrix(s)))
    for n in neighbours:
        print 'Energy of Neighbour: ' + str(np.matrix(n)) + ' = ' + str(energy(np.matrix(n)))
    print ' '

## Running with inhibition constant of 0.5 gives:
# (0, 0, 0, 0, 0, 0, 0, 0, 0) - All neighbours have energy 0
# (0, 0, 0, 1, 1, 1, 0, 0, 0) - Not minima
# (0, 1, 0, 0, 1, 0, 0, 1, 0) - Not minima
# (0, 1, 0, 1, 1, 1, 0, 1, 0) - Is minima with -8

## Running with inhibition constant of 1 gives
# (0, 0, 0, 0, 0, 0, 0, 0, 0) - All neighbours have energy 0
# (0, 0, 0, 1, 1, 1, 0, 0, 0) - Is minima with -6
# (0, 1, 0, 0, 1, 0, 0, 1, 0) - Is minima with -6

### Q3
# For each of the 20 initial state in 'init', run the network asynchronously
print "Asynchronous update:"
for s in init:
    print str(s) + ': ' + str(hopfield(weight_matrix, np.matrix(s), async_update, 100))

print ' '
# For each of the 20 initial state in 'init', run the network synchronously
print "Synchronous update:"
for s in init:
    print str(s) + ': ' + str(hopfield(weight_matrix, np.matrix(s), sync_update, 100))


# No, network does not always converge to the same fixed point given an initial state

## Running with inhibition constant of 0.5 gives:
# Initial states converge to one of the the 4 fixed Points

## Running with inhibition constant of 1 gives
# Initial states converge to one of the 3 fixed points
