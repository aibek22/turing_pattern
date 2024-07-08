import pandas as pd
import numpy as np
import os
from scipy.signal import argrelextrema
import time
import multiprocessing
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots
import matplotlib.pyplot as plt

def gen_matrix_F(n, vr):
    I = np.eye(n)
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)
    np.fill_diagonal(B, 0)
    A = -I + B
    D = np.zeros((n, n))
    return A, D

def process_tuple(tuplet):
    p, x, y = tuplet
    D = gen_matrix_F(p[0], p[1])[1]
    D[0, 0] = 10**x
    D[1, 1] = 10**y
    k = np.arange(0, 10, 0.01)

    t1a = []
    t1b = []
    t2a = []
    t2b = []

    for _ in range(1000):  # 1,000 matrices
        m = gen_matrix_F(p[0], p[1])[0]
        ev = np.linalg.eigvals(m)
        if np.max(np.real(ev)) < 0:  # if matrix is stable
            Em = []
            Emi = []
            for ki in k:
                R = m - D * (ki ** 2)
                eigval = np.linalg.eigvals(R)
                Em.append(np.max(np.real(eigval)))
                idx_max = np.argmax(np.real(eigval))
                Emi.append(np.imag(eigval[idx_max]))
            a = np.max(Em)
            index = np.argmax(Em)
            nEm = np.array(Em)
            if a > 0:
                if Emi[index] == 0:
                    numZeroCrossing = np.count_nonzero(np.diff(np.sign(Em)))  # Count zero crossings
                    numpositivelocalmaxima = np.sum(nEm[argrelextrema(nEm, np.greater)] > 0) > 0
                    if numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 0:
                        t1a.append(m)
                    elif numpositivelocalmaxima > 0 and numZeroCrossing == 1:
                        t1b.append(m)
                    elif numpositivelocalmaxima == 0 and numZeroCrossing % 2 == 1:
                        t2a.append(m)
                    elif numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 1:
                        t2b.append(m)
    percent = (len(t1a) + len(t1b)) * 0.1
    return percent

def main():
    step = 0.1
    dx = np.arange(-3, 3 + step, step).tolist()
    dy = np.arange(-3, 3 + step, step).tolist()

    parameters = [(3, 1), (4, 1), (10, 0.2), (50, 0.02)]

    dp_list = [(par, x, y) for par in parameters for x in dx for y in dy]

    start_time = time.time()

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    # Monitor progress
    results = []
    total_tasks = len(dp_list)
    for i, result in enumerate(pool.imap(process_tuple, dp_list), 1):
        results.append(result)
        if i % 100 == 0 or i == total_tasks:
            print(f'Processed {i}/{total_tasks} tasks... ({(i/total_tasks)*100:.2f}%)')

    pool.close()
    pool.join()

    end_time = time.time()

    df_data = pd.DataFrame({'N': [x[0][0] for x in dp_list], 'Dx': [x[1] for x in dp_list], 'Dy': [x[2] for x in dp_list], 'Percentage': results})
    df_data.to_csv(os.path.join('./', 'heatmap_fig8.csv'), index=False)
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    path = './heatmap_fig8.csv'
    data = pd.read_csv(path)
    data.head()

    df_n3 = data[data['N'] == 3]
    df_n4 = data[data['N'] == 4]
    df_n10 = data[data['N'] == 10]
    df_n50 = data[data['N'] == 50]

    per3 = np.array(df_n3['Percentage'])
    l = len(dx)
    percentage3 = [per3[i:i + l] for i in range(0, len(per3), l)]

    per4 = np.array(df_n4['Percentage'])
    percentage4 = [per4[i:i + l] for i in range(0, len(per4), l)]

    per10 = np.array(df_n10['Percentage'])
    percentage10 = [per10[i:i + l] for i in range(0, len(per10), l)]

    per50 = np.array(df_n50['Percentage'])
    percentage50 = [per50[i:i + l] for i in range(0, len(per50), l)]

    dfs = [percentage3, percentage4, percentage10, percentage50]
    titles = ['N = 3', 'N = 4', 'N = 10', 'N = 50']

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for i in range(4):
        row = i // 2
        col = i % 2
        Y, X = np.meshgrid(dy, dx)
        ax = axs[row, col]
        colormap = ax.pcolormesh(X, Y, dfs[i], cmap='viridis')
        ax.set_title(titles[i], fontweight='bold', style='italic')
        ax.plot(X, X, color='white', linestyle='--')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    cbar_ax = fig.add_axes([0.95, 0.1, 0.03, 0.36])
    cbar = fig.colorbar(colormap, ax=axs, shrink=0.5, cax=cbar_ax)
    fig.text(0.046, 0.5, r'$log_{10}$$\it{D_{y}}$', va='center', fontsize=22, rotation='vertical', fontweight='bold')
    fig.text(0.51, 0.03, r'$log_{10}$$\it{D_{x}}$', ha='center', fontsize=22, fontweight='bold')
    plt.savefig('heatmap_fig8.png')
    print("Heatmap figure saved as 'heatmap_fig8.png'")
    plt.show()

if __name__ == '__main__':
    main()
