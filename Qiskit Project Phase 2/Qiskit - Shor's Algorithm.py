#!/usr/bin/env python
# coding: utf-8

# ## Necessary Imports

# In[8]:


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

# In[9]:


num_qubits = 2
runtimes = []


# ### Letting the user decide as to how many qubits is to be analyzed

# In[11]:


# To decide maximum number of qubits we need to analyze the run time
max_qubits = int(input("Enterthe maximum number of qubits for which you want to analyze algorithm"))


# ## Shor's Algorithm

# In[ ]:


max_qubits=0
def c_amodn(a, power, n, num_qubits):
    """Controlled multiplication by a mod n"""
    U = QuantumCircuit(num_qubits)
    U.unitary(controlled_modular_exponentiation_gate(a, power, n, num_qubits), range(num_qubits), label=f"{a}^{{{2**power}}} mod {n}")
    c_U = U.control()
    return c_U

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT Dagger"
    return qc

def controlled_modular_exponentiation_gate(a, power, n, num_qubits):
    U = np.eye(2 ** num_qubits, dtype=complex)
    for x in range(2 ** num_qubits):
        y = (x * a ** (2 ** power)) % n
        if x != y:
            temp = np.copy(U[x, :])
            U[x, :] = U[y, :]
            U[y, :] = temp
    return UnitaryGate(U, label=f"{a}^{{{2**power}}} mod {n}")

qubit_range = range(num_qubits, max_qubits + 1)
number_of_qubits = []
memory_values_shors = []
time_values_shors = []

n =  int(input("Enter n value: ")) # Take input for 'n'
binary_num = bin(n)[2:]  

a_min = 2  # the minimum value of a to try
a_max = n - 1  # the maximum value of a to try

# choose a random value of a between a_min and a_max
a = random.randint(a_min, a_max)

# a = int(input("Enter a value between 1 - entered n value")) # Take input for 'a'
num_qubits = len(binary_num)
max_qubits = 2*int(num_qubits) + 3

for n_count in range(num_qubits, max_qubits + 1):
    number_of_qubits.append(n_count)
    total_time_inner=[]
    memory_usage_inner=[]
    for _ in range(0,10):
        tracemalloc.start()
        qc = QuantumCircuit(n_count + num_qubits, n_count)

        for q in range(n_count):
            qc.h(q)

        qc.x(3 + n_count)

        for q in range(n_count):
            qc.append(c_amodn(a, q, n, num_qubits), [q] + [i + n_count for i in range(num_qubits)])

        qc.append(qft_dagger(n_count), range(n_count))

        qc.measure(range(n_count), range(n_count))

        backend = Aer.get_backend('qasm_simulator')
        start_time = time.time()
        results_shors = execute(qc, backend, shots=1024).result()
        end_time = time.time()
        mem_usage = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        tracemalloc.stop()
        time_taken=end_time-start_time
        memory_usage_inner.append(mem_usage)
        total_time_inner.append(time_taken)
    print(f"No. of Qubit: {n_count} -- Execution time: {(np.mean(total_time_inner))} s")
    time_values_shors.append(np.mean(total_time_inner)) 
    memory_values_shors.append(np.mean(memory_usage_inner))


# ### Plotting memory graph for shors

# In[ ]:


# Plot the curve
plt.plot(number_of_qubits, memory_values_shors,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Shors Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for shors

# In[ ]:


# Plot the curve
plt.plot(number_of_qubits, time_values_shors,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Execution time (in sec)")
plt.title("Shors Time Plot")

# Show the plot
plt.show()


# ### Histogram of the Shor's algorithm

# In[ ]:


#get the counts and plot histogram
counts=results_shors.get_counts(qc)
plot_histogram(counts,figsize=(3,4))


# In[ ]:




