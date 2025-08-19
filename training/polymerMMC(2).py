import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge, physical_constants
from math import dist
from scipy.spatial.distance import cdist
# 
###################################################################################################

initial_microstate = 'restart'
#initial_microstate = 'new'

###################################################################################################

N = 22  # number of atoms

###################################################################################################

temperature = 100.0

###################################################################################################

nAttempts = 5000000 #number of attempts to perform

###################################################################################################

dscale = 0.05  # VERY IMPORTANT: it sets the range [-d/2,d/2] of the random displacements of the atoms.

###################################################################################################

printerval = 50000 
xyz_printerval = 1000
printskip = 10000 // printerval

###################################################################################################

center = N // 2 

###################################################################################################

bond_length = 1.53
d = dscale*bond_length 

sigma = 4.5
epsilon = 0.00485678
k = 15.18

####################################################################################################

kB_SI = Boltzmann
kB, kB_unit = physical_constants['Boltzmann constant in eV/K'][0:2]
target_kT = temperature*kB  # temperature

####################################################################################################
#
# Create the files

allstepsfile = 'MMC_data.csv'

onestepfile = 'MMC_step_data.csv'

with open("structure.xyz", "w") as f:
    pass

####################################################################################################
def read_xyz(filepath):
    with open(filepath, 'r') as frestart:
        # read the number of atoms
        num_atoms = int(frestart.readline())
        next(frestart)

        # initialize an empty list to store the coordinates
        coordinates = []

        # loop through the file and read the coordinates
        for line in frestart:
            # split the line by whitespace
            elements = line.split()

            # convert the elements to floats
            x, y, z = map(float, elements[1:])

            # append the coordinates to the list
            coordinates.append([x, y, z])

    # convert the list to a NumPy array
    coordinates = np.array(coordinates)

    return coordinates
####################################################################################################

####################################################################################################

coords = np.zeros((N, 3))

#

if initial_microstate == 'restart':
   coords = read_xyz('restart.xyz')
   print("\n Starting from a restart file.\n")
else:
   print("\n Starting from a random structure.\n")
   coords[0] = coords[0] + (np.random.rand(3) - 0.5) * d
   for i in range(1, N):
      coords[i,0] = coords[i - 1,0] + bond_length 
      coords[i] = coords[i] + (np.random.rand(3) - 0.5) * d

with open("structure-%d.xyz" % 0, "w") as f:
            f.write("%d\n\n" % N)
            for j in range(N):
                f.write("C %f %f %f\n" % (coords[j, 0], coords[j, 1], coords[j, 2]))
#
####################################################################################################
#
bondmatrix = np.zeros((N,N))
for i in range(N):
    if i != 0: bondmatrix[i,i-1] = bond_length
    if i != N-1: bondmatrix[i,i+1] = bond_length
####################################################################################################
#
#
def scipy_cdist(X,Y):
    return cdist(X,Y,metric='euclidean')

distances = scipy_cdist(coords,coords)
distances -= bondmatrix

energy = 0
for i in range(N-1): 
    energy += 0.5*k*distances[i,i+1]**2
    for j in range(i+3,N):
        sixterm = (sigma/distances[i,j])**6.0
        energy +=  epsilon * (sixterm**2 - 2*sixterm)


####################################################################################################
#
# initialize lists to store data

attempts = []
PE = []
L2N = []
PEave = []
L2Nave = []

print_format = " {:6d}"+"{:9.1f}"+" "+"{:8.2f}"+7*" {:10.6f}"+2*" {:8.4f}"
prstep = 0
prattempts = []

####################################################################################################
#
# perform Metropolis Monte Carlo

accepted_moves = 0
rejected_moves = 0
#
for i in range(nAttempts):
    # select random atom
    atom = random.randint(0, N - 1)
    
    # store current coordinates
    current_coords = coords[atom].copy()
    
    # generate new coordinates
    new_coords = current_coords + (np.random.rand(3) - 0.5) * d 
    newdist = scipy_cdist(new_coords.reshape((1,3)),coords).reshape(N) - bondmatrix[atom]

# the atom^th cpt of newdist is not quite zero now
    olddist = distances[atom]
#
    # calculate energy difference
    delta_E = 0
    if atom != 0: delta_E += 0.5*k*(newdist[atom-1]-olddist[atom-1])*(newdist[atom-1]+olddist[atom-1])
    if atom != N-1: delta_E += 0.5*k*(newdist[atom+1]-olddist[atom+1])*(newdist[atom+1]+olddist[atom+1])

    for j in range(N):
        if abs(j - atom) > 2:
            old_sixterm = (sigma/olddist[j])**6.0
            sixterm = (sigma/newdist[j])**6.0
            delta_E +=  epsilon * ((sixterm**2 - 2*sixterm) - (old_sixterm**2 - 2*old_sixterm))

    # accept or reject new coordinates
    if delta_E <= 0:
        accept = True
    elif random.random() < np.exp(-delta_E / target_kT):
        accept = True
    else:
        accept = False

    # store data
    if accept:
       accepted_moves += 1
       coords[atom] = new_coords
       distances[atom] = newdist
       distances[:,atom] = newdist.reshape(1,N)
       energy += delta_E
    else:
       rejected_moves += 1

    attempts.append(i)
    PE.append(energy/N)
    L2N.append(distances[0,N-1]**2.0/N)


    energies_printerval = 10000
    # Append the data to the file energies.csv:
    if i % energies_printerval == 0 or i == nAttempts-1:
       with open("energies.csv", "a") as f:
          f.write("%d, %f, %f\n" % (i, PE[-1], L2N[-1]))

#    # Append the data to the file energies.csv:
#    with open("energies.csv", "a") as f:
#        f.write("%d, %f, %f\n" % (i, PE[-1], L2N[-1]))

    # Print structure to file structure.xyz:
    if i % xyz_printerval == 0 or i == nAttempts-1: # every xyz_printerval attempts
        with open("structure.xyz", "a") as f:
            f.write("%d\n\n" % N)
            for j in range(N):
               f.write("C %f %f %f\n" % (coords[j, 0]-coords[center,0], coords[j, 1]-coords[center,1], coords[j, 2]-coords[center,2])) 
    
    # Print running averages to screen and MMC_step_data.csv:
    if i % printerval == 0 or i == nAttempts-1: # every printerval attempts
       prattempts.append(i)
       PEave.append(sum(PE)/len(PE))
       L2Nave.append(sum(L2N)/len(L2N)) 
       print(" Attempt {:8d}, PEave = {:14.10f}, L2ave/N = {:10.4f}".format(i, PEave[-1], L2Nave[-1]))
       with open(onestepfile,"a") as f:
            f.write("%d, %f, %f\n" % (prattempts[-1], PEave[-1], L2Nave[-1]))
       prstep += 1

####################################################################################################

print("\n Finished !")
print("\n {:8d} moves were accepted and {:8d} moves were rejected.".format(accepted_moves,rejected_moves))

# Print restart file:
with open("restart.xyz", "w") as f:
     f.write("%d\n\n" % N)
     for j in range(N):
         f.write("C %f %f %f\n" % (coords[j, 0]-coords[center,0], coords[j, 1]-coords[center,1], coords[j, 2]-coords[center,2])) 

# Print data to file MMC_data.csv:
with open(allstepsfile,"a") as f:
     f.write("%f, %d, %d,  %d, %d, %f, %f\n" % (temperature, N, prattempts[-1], accepted_moves, rejected_moves, PEave[-1],  L2Nave[-1]))

# Plot the data
# Create a 2x2 panel of subplots
fig, axs = plt.subplots(2,2, sharex='col',  figsize=(20,10))


# Plot the data in the first subplot
axs[0,0].plot(attempts, PE,'tab:blue')
axs[0,0].plot(prattempts, PEave,'tab:red')
axs[0,0].set_ylabel('PE',fontsize=20)

axs[0,1].plot(prattempts[printskip:], PEave[printskip:],'tab:red')
axs[0,1].set_ylabel('Running average of PE',fontsize=16)

# Plot the data in the second subplot

axs[1,0].plot(attempts, L2N,'tab:blue')
axs[1,0].plot(prattempts, L2Nave,'tab:red')
axs[1,0].set_xlabel('Attempt number',fontsize=20)
axs[1,0].set_ylabel('L2N',fontsize=20)

axs[1,1].plot(prattempts[printskip:], L2Nave[printskip:],'tab:red')
axs[1,1].set_xlabel('Attempt number',fontsize=20)
axs[1,1].set_ylabel('Running average of L2N',fontsize=16)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)

fig.suptitle('T = %i K' %temperature,fontsize=30)
# Show the plots
plt.show()

 
