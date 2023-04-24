#!/usr/bin/env python
# coding: utf-8

# ## Necessary Imports

# In[2]:


from qiskit import QuantumCircuit, Aer, execute, transpile, assemble
from qiskit.circuit.library.standard_gates import U1Gate
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

# In[3]:


num_qubits = 2
runtimes = []


# ### Letting the user decide as to how many qubits is to be analyzed

# In[4]:


# To decide maximum number of qubits we need to analyze the run time
max_qubits = int(input("Enterthe maximum number of qubits for which you want to analyze algorithm"))


# ## Bernstein-Vazirani Algorithm

# In[ ]:


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


# In[ ]:


number_of_qubits = []
memory_values = []
time_values = []
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
    time_values.append(np.mean(total_time_inner)) 
    memory_values.append(np.mean(memory_usage_inner))


# ### Plotting memory graph for bernstein vazirani

# In[ ]:


# print(number_of_qubits, memory_values)
# Plot the curve
plt.plot(number_of_qubits, memory_values,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Bernstein-Vazirani Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for bernstein vazirani

# In[ ]:


#print(number_of_qubits, time_values)
# Plot the curve
plt.plot(number_of_qubits, time_values,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Execution time (in sec)")
plt.title("Bernstein-Vazirani Time Plot")

# Show the plot
plt.show()


# ### Histogram of the Bernstein - Vazirani algorithm

# In[ ]:


#get the counts and plot histogram
print(hidden_string)
counts=result.get_counts(qc)
plot_histogram(counts,figsize=(3,4))


# ## Grover's Algorithm

# #### Question.
# ##### In Grover’s alg., the number of iterations K is in the order of sqrt(N). That is, K = c . sqrt(N). Identify what c is in your experiments**
# 
# #### Answer 
# ##### In the code below, the value of c is 2 i.e. number of iterations.
# ##### However if we want to calculate the value of c from the code, we can use formula
# ##### "c = math.sqrt(N/M) and then assign iterations = c"
# ##### In the code below, N is 2^n number of qubits and then M is len(solutions) array = 4 in this case

# In[ ]:


qubit_range = range(num_qubits, max_qubits+1)
number_of_qubits = []
memory_values_grovers = []
time_values_grovers = []

for n in range(num_qubits, max_qubits+1):
    total_time_inner=[]
    memory_usage_inner=[]
    number_of_qubits.append(n)
    # Define the function f as a list of indices of the solutions
    solutions = [2, 3]

    # Define the number of qubits and the number of iterations of the algorithm
    n = n
    iterations = math.ceil(math.sqrt(2**n/2))
    for _ in range(0,10):
        tracemalloc.start()
        # Create the quantum circuit
        qc = QuantumCircuit(n, n)

        # Apply Hadamard gates to all qubits
        qc.h(range(n))

        # Apply the Grover iteration for the desired number of iterations
        for i in range(iterations):
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
        shots = 1024
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

# In[ ]:


# Plot the curve
plt.plot(number_of_qubits, memory_values_grovers,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Grovers Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for grovers

# In[ ]:


# Plot the curve
plt.plot(number_of_qubits, time_values_grovers,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Execution time (in sec)")
plt.title("Grovers Time Plot")

# Show the plot
plt.show()


# ### Histogram of the Grover's algorithm

# In[ ]:


#get the counts and plot histogram
counts=result_grovers.get_counts(qc)
plot_histogram(counts,figsize=(3,4))


# ## Shor's Algorithm

# In[5]:


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

# In[6]:


# Plot the curve
plt.plot(number_of_qubits, memory_values_shors,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Shors Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for shors

# In[7]:


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


# ## Simon's Algorithm

# In[ ]:


number_of_qubits = []
memory_values_simon = []
time_values_simon = []

for n_count in range(num_qubits, max_qubits+1):
    total_time_inner=[]
    memory_usage_inner=[]
    number_of_qubits.append(n_count)
    for _ in range(0,10):
        # Define the number of qubits and the secret string for Simon's algorithm
        tracemalloc.start()

       # Define the function f
        n = n_count  # Number of bits in the input
        s = ''.join([random.choice(['0', '1']) for _ in range(n)])  # Hidden bitstring
        f = lambda x: '{0:b}'.format(int(x, 2) ^ int(x, 2) & int(s, 2)).zfill(n)  # XOR with s

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
        shots = 1024
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

# In[ ]:


# Plot the curve
plt.plot(number_of_qubits, memory_values_simon,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Memory Consumption (in MB)")
plt.title("Simons Memory Plot")

# Show the plot
plt.show()


# ### Plotting time graph for simon's

# In[ ]:


# Plot the curve
plt.plot(number_of_qubits, time_values_simon,marker='.')

# Add labels to the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Execution time (in sec)")
plt.title("Simons Time Plot")

# Show the plot
plt.show()


# ### Histogram of the Simon algorithm

# In[ ]:


#get the counts and plot histogram
counts=result_simon.get_counts(qc)
plot_histogram(counts,figsize=(3,4))

