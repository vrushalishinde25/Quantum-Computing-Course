#!/usr/bin/env python
# coding: utf-8

# ## Necessary Imports

# In[1]:


from qiskit import QuantumCircuit, Aer, execute, transpile, assemble
from qiskit.circuit.library.standard_gates import U1Gate
from qiskit.extensions import UnitaryGate
import matplotlib.pyplot as plt
import tracemalloc
import random
import numpy as np
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from math import pi, sqrt, ceil, gcd, isqrt
import time
import math

# for readymade inbuilt shors
from qiskit.algorithms import Shor #previousy it was from qiskit.aqua but got deprecated after qiskit 0.27.x


# ### Declaring the minimum number of qubits as two

# In[2]:


num_qubits = 2
runtimes = []


# ### Letting the user decide as to how many qubits is to be analyzed

# In[3]:


# To decide maximum number of qubits we need to analyze the run time
max_qubits = int(input("Enterthe maximum number of qubits for which you want to analyze algorithm"))


# ## Simon's Algorithm

# In[4]:


number_of_qubits = []
memory_values_simon = []
time_values_simon = []
max_qubits = 100
for n_count in range(num_qubits, max_qubits+1):
    total_time_inner=[]
    memory_usage_inner=[]
    number_of_qubits.append(n_count)
    # Define the function f
    n = n_count  # Number of bits in the input
    s = ''.join([random.choice(['0', '1']) for _ in range(n)])  # Hidden bitstring
    f = lambda x: '{0:b}'.format(int(x, 2) ^ int(x, 2) & int(s, 2)).zfill(n)  # XOR with s    
    
    for _ in range(0,10):
        # Define the number of qubits and the secret string for Simon's algorithm
        tracemalloc.start()
        # Create the quantum circuit
        qc = QuantumCircuit(n*2, n)

        # Apply Hadamard gates to the first n qubits
        for i in range(n):
            qc.h(i)

        # Apply the function f
        for i in range(n):
            qc.cx(i, n + int(f('{0:b}'.format(i).zfill(n)), 2))

        # Apply Hadamard gates to the first n qubits again
        for i in range(n):
            qc.h(i)

        # Measure the first n qubits
        for i in range(n):
            qc.measure(i, i)

        # Run the circuit on the qasm simulator
        backend = Aer.get_backend('qasm_simulator')
        shots = 2048
        start_time = time.time()
        result_simon = execute(qc, backend, shots=shots).result()
        end_time = time.time()
        # Get the current size of traced memory
        mem_usage = tracemalloc.get_traced_memory()[0] / 1024 / 1024 # convert to MB

        # Stop tracing memory allocations
        tracemalloc.stop()
        time_taken=end_time-start_time
        memory_usage_inner.append(mem_usage)
        total_time_inner.append(time_taken)
    print(f"No. of Qubit: {n} -- Execution time: {(np.mean(total_time_inner))} s")
    time_values_simon.append(np.mean(total_time_inner)) 
    memory_values_simon.append(np.mean(memory_usage_inner))


# ### Plotting memory graph for simon's

# In[5]:


# Plot the curve
plt.plot(number_of_qubits, memory_values_simon,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Simons Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for simon's

# In[6]:


# Plot the curve
plt.plot(number_of_qubits, time_values_simon,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Execution time (in sec)")
plt.title("Simons Time Plot")

# Show the plot
plt.show()


# ### Histogram of the Simon algorithm

# In[7]:


#get the counts and plot histogram
counts=result_simon.get_counts(qc)
plot_histogram(counts,figsize=(3,4))


# In[ ]:




