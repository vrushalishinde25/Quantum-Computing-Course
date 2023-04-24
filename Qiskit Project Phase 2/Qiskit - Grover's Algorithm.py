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


# ## Grover's Algorithm

# #### Question.
# ##### In Grover’s alg., the number of iterations K is in the order of sqrt(N). That is, K = c . sqrt(N). Identify what c is in your experiments**
# 
# #### Answer 
# ##### In the code below, the value of c is 2 i.e. number of iterations.
# ##### However if we want to calculate the value of c from the code, we can use formula
# ##### "c = math.sqrt(N/M) and then assign iterations = c"
# ##### In the code below, N is 2^n number of qubits and then M is len(solutions) array = 4 in this case
# ##### This calculation is derived from the formula K = c * sqrt(N), with N = 2^n, and c = 1/2. The ceil function is used to round up the result to the nearest integer value. This means that in this experiment, the constant coefficient c is equal to 1/2, which is used to calculate the number of Grover iterations based on the number of qubits.
# 
# ##### This will make the number of iterations change depending on the size of the search space and make the algorithm's behavior closer to the theoretical expectation of Grover's algorithm.
# ##### In the code below, N is 2^n number of qubits and then M is len(solutions) array = 4 in this case

# In[4]:


max_qubits =25
qubit_range = range(num_qubits, max_qubits+1)
number_of_qubits = []
memory_values_grovers = []
time_values_grovers = []
itr=[]
c=[]
for n in range(num_qubits, max_qubits+1):
    total_time_inner=[]
    memory_usage_inner=[]
    number_of_qubits.append(n)
    # Define the function f as a list of indices of the solutions
    solutions = [2, 3,5,7]


    # Define the number of qubits and the number of iterations of the algorithm
    N=2**n
    M=len(solutions)
    k = round(math.pi/4 * math.sqrt(N/M) - 1/2)
    c_k = round(math.pi/4 * math.sqrt(N/M) - 1/2) / sqrt(N)
    itr.append(k)
    c.append(c_k)
    
    for _ in range(0,10):
        tracemalloc.start()
        # Create the quantum circuit
        qc = QuantumCircuit(n, n)

        # Apply Hadamard gates to all qubits
        qc.h(range(n))

        # Apply the Grover iteration for the desired number of iterations
        for i in range(k):
            # Oracle
            for solution in solutions:
                # Convert the solution index to a binary string and pad with zeros
                b = bin(solution)[2:].zfill(n)
                # Apply X gates to the qubits corresponding to a 1 in the binary string
                for j in range(n):
                    if b[j] == '1':
                        qc.x(j)
                # Apply a phase flip to the state corresponding to the solution
                qc.cz(0, n-1)
                # Apply X gates again to the qubits corresponding to a 1 in the binary string
                for j in range(n):
                    if b[j] == '1':
                        qc.x(j)
            # Diffusion operator
            qc.h(range(n))
            qc.x(range(n))
            qc.h(n-1)
            qc.mct(list(range(n-1)), n-1)
            qc.h(n-1)
            qc.x(range(n))
            qc.h(range(n))

        # Measure all qubits
        qc.measure(range(n), range(n))

        # Execute the circuit on a simulator
        backend = Aer.get_backend('qasm_simulator')
        shots = 2048
        start_time = time.time()
        result_grovers = execute(qc, backend=backend, shots=shots,memory=True).result()
        end_time = time.time()
        # Get the current size of traced memory
        mem_usage = tracemalloc.get_traced_memory()[0] / 1024 / 1024 # convert to MB

        # Stop tracing memory allocations
        tracemalloc.stop()
        time_taken=end_time-start_time
        memory_usage_inner.append(mem_usage)
        total_time_inner.append(time_taken)
        
    print(f"No. of Qubit: {n} -- Execution time: {(np.mean(total_time_inner))} s")
    time_values_grovers.append(np.mean(total_time_inner)) 
    memory_values_grovers.append(np.mean(memory_usage_inner))


# ### Plotting memory graph for grovers

# In[5]:


# Plot the curve
plt.plot(number_of_qubits, memory_values_grovers,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Grovers Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for grovers

# In[6]:


# Plot the curve
plt.plot([i + 4 for i in range(len(time_values_grovers))], time_values_grovers,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Execution time (in sec)")
plt.title("Grovers Time Plot")

# Show the plot
plt.show()


# # Plot number of iteration vs Number of Qubits

# In[7]:


# Plot the curve
plt.plot([i + 4 for i in range(len(time_values_grovers))], c,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("c")
plt.title("c vs. No. of Qubits")

# Show the plot
plt.show()


# In[8]:


# Plot the curve
plt.plot([i + 4 for i in range(len(time_values_grovers))], itr,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Number of Interations (K)")
plt.title("K vs. No. of Qubits")

# Show the plot
plt.show()


# In[9]:


# Plot the curve
plt.plot(itr,c,marker='.')

# Add labels to the plot
plt.xlabel("K")
plt.ylabel("c")
plt.title("K vs. c")

# Show the plot
plt.show()


# ### Histogram of the Grover's algorithm

# In[10]:


#get the counts and plot histogram
counts=result_grovers.get_counts(qc)
plot_histogram(counts,figsize=(3,4))


# In[ ]:




