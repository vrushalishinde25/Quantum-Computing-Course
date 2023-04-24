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


# ### Declaring the minimum number of qubits as two

# In[2]:


num_qubits = 2
runtimes = []


# ### Letting the user decide as to how many qubits is to be analyzed

# In[3]:


# To decide maximum number of qubits we need to analyze the run time
max_qubits = int(input("Enterthe maximum number of qubits for which you want to analyze algorithm"))


# ## Bernstein-Vazirani Algorithm

# In[4]:


# Define the oracle function that encodes a hidden bitstring
def oracle(n, hidden_string):
    qc = QuantumCircuit(n+1)
    for i, char in enumerate(reversed(hidden_string)):
        if char == '1':
            qc.cx(i, n)
    return qc

# Define the Bernstein-Vazirani algorithm
def bernstein_vazirani(n, hidden_string):
    # Create a quantum circuit with n+1 qubits and n classical bits
    qc = QuantumCircuit(n+1, n)
    # Apply a Hadamard gate to all qubits except the last one
    qc.h(range(n)) 
    # Apply the oracle function
    qc.append(oracle(n, hidden_string), range(n+1))  
    # Apply a Hadamard gate to all qubits except the last one again
    qc.h(range(n))   
    # Measure all qubits except the last one
    qc.measure(range(n), range(n))  
    return qc


# In[5]:


number_of_qubits = []
memory_values_bv = []
time_values_bv = []
max_qubits=100
for i in range(num_qubits, max_qubits+1):
    number_of_qubits.append(i)
    n = i  # number of bits in the hidden string
    total_time_inner=[]
    memory_usage_inner=[]
    hidden_string = ''.join([random.choice(['0', '1']) for _ in range(n)])
    for _ in range(0,10):
        
        tracemalloc.start()
        qc = bernstein_vazirani(i, hidden_string)
        
        backend = Aer.get_backend('qasm_simulator')
        start_time = time.time()
        result = execute(qc, backend, shots=2048, memory=True).result()
        end_time = time.time()
        # Get the current size of traced memory
        mem_usage = tracemalloc.get_traced_memory()[0] / 1024 / 1024 # convert to MB

        # Stop tracing memory allocations
        tracemalloc.stop()
       
        time_taken=end_time-start_time
        memory_usage_inner.append(mem_usage)
        total_time_inner.append(time_taken)
        
    print(f"No. of Qubit: {n} -- Execution time: {(np.mean(total_time_inner))} s")
    time_values_bv.append(np.mean(total_time_inner)) 
    memory_values_bv.append(np.mean(memory_usage_inner))


# ### Plotting memory graph for bernstein vazirani

# In[6]:


# print(number_of_qubits, memory_values)
# Plot the curve
plt.plot(number_of_qubits, memory_values_bv,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Bernstein-Vazirani Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for bernstein vazirani

# In[7]:


#print(number_of_qubits, time_values)
# Plot the curve
plt.plot(number_of_qubits, time_values_bv,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Execution time (in sec)")
plt.title("Bernstein-Vazirani Time Plot")

# Show the plot
plt.show()


# In[ ]:




