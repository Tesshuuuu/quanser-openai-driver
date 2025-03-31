import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('quanser_qube.pkl', 'rb') as f:
    data = pickle.load(f)

all_Ks = data['all_Ks']
all_SA_cost = data['all_SA_cost']

SA_error = np.array(all_SA_cost[0][:]) - np.array(all_SA_cost[0][-1])

plt.figure(figsize=(10, 5))
iterations = np.arange(len(all_SA_cost[0]))  
plt.plot(iterations, SA_error)
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel(r'$J_{SA}(K) - J_{SA}(K^*_{SA})$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.savefig('SA_error.png')
plt.show()
