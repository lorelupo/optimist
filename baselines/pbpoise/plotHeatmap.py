import seaborn as sns
import numpy as np

print("Loading files")
ess1 = np.genfromtxt('ess_version1.csv', delimiter=',')
ess2 = np.genfromtxt('ess_version2.csv', delimiter=',')
n_def1 = np.genfromtxt('n_def_version1.csv', delimiter=',')
n_def2 = np.genfromtxt('n_def_version2.csv', delimiter=',')

average = np.mean(ess1)
sns.heatmap(ess1, center=average)
