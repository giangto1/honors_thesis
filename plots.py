import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')

lam = [0.01, 0.1, 1, 10, 100]
# Experiment 1
yu=  [251.57542252540588, 254.40057754516602, 262.5241198539734, 269.13786482810974, 266.12157249450684]
cvx=  [11983.64830493927, 159.09524822235107, 106.61818313598633, 157.16452741622925, 136.4798984527588]

plt.rcParams['text.usetex'] = False
plt.figure()
plt.semilogx(lam,yu, label='Yu et al. (2024)')
plt.semilogx(lam,cvx, label='CVX custom solver')  # Second line

# Add labels and legend
plt.xlabel('lambda')
plt.ylabel('runtime')
plt.title('Runtime Efficiency Comparison for num nodes = 50, p = 50')

plt.legend()
plt.xticks(lam)
plt.grid(True)
# plt.show()

plt.savefig("e1.png")

# Experiment 2
yu =  [5449.582875967026, 5754.665199756622, 2038.4785087108612, 5323.783842802048, 5332.7773768901825]
cvx =  [21.85471796989441, 122.54377388954163, 1227.7022109031677, 1415.2817840576172, 1384.7840945720673]

plt.rcParams['text.usetex'] = False
plt.figure()
plt.semilogx(lam,yu, label='Yu et al. (2024)')
plt.semilogx(lam,cvx, label='CVX custom solver')  # Second line

# Add labels and legend
plt.xlabel('lambda')
plt.ylabel('runtime')
plt.title('Runtime Efficiency Comparison for num nodes = 300, p = 50')

plt.legend()
plt.xticks(lam)
plt.grid(True)
# plt.show()

plt.savefig("e2.png")

# Experiment 3
yu =  [410.94210028648376, 403.22693634033203, 8.937906742095947, 14.562551021575928, 5.666273355484009]
cvx =  [97.41865968704224, 52.548227071762085, 3.634629249572754, 0.7530720233917236, 0.6616101264953613]

plt.rcParams['text.usetex'] = False
plt.figure()
plt.semilogx(lam,yu, label='Yu et al. (2024)')
plt.semilogx(lam,cvx, label='CVX custom solver')  # Second line

# Add labels and legend
plt.xlabel('lambda')
plt.ylabel('runtime')
plt.title('Runtime Efficiency Comparison for num nodes = 50, p = 5')

plt.legend()
plt.xticks(lam)
plt.grid(True)
# plt.show()

plt.savefig("e3.png")

# Experiment 4
yu =  [2348.4292962551117, 1601.436512708664, 118.50129580497742, 64.4115982055664, 84.40664649009705]
cvx =  [603.7811045646667, 294.67011857032776, 47.92143201828003, 638.547860622406, 2.4812986850738525]

plt.rcParams['text.usetex'] = False
plt.figure()
plt.semilogx(lam,yu, label='Yu et al. (2024)')
plt.semilogx(lam,cvx, label='CVX custom solver')  # Second line

# Add labels and legend
plt.xlabel('lambda')
plt.ylabel('runtime')
plt.title('Runtime Efficiency Comparison for num nodes = 300, p = 5')

plt.legend()
plt.xticks(lam)
plt.grid(True)
# plt.show()

plt.savefig("e4.png")