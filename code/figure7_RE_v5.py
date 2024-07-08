import pandas as pd
import numpy as np
import os
from scipy.signal import argrelextrema
import time
import multiprocessing
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots
import matplotlib.pyplot as plt
import seaborn as sns

def gen_matrix_F(n, vr):
    I = np.eye(n)
    mu = 0
    B = mu + np.sqrt(vr) * np.random.randn(n, n)
    np.fill_diagonal(B, 0)
    A = -I + B
    D = np.zeros((n, n))
    return A, D

def process_tuple(tuplet):
    n, vr = tuplet
    rm = []
    for _ in range(1000):
        m = gen_matrix_F(n, vr)[0]
        rm.append(m)

    srm = []
    for m in rm:
        ev = np.linalg.eigvals(m)
        if np.max(np.real(ev)) < 0:
            srm.append(m)
    
    D = gen_matrix_F(n, vr)[1]
    D[0, 0] = 1
    D[1, 1] = 100
    k = np.arange(0, 10, 0.01)  
  
    t1a = []
    t1ar = []
    t1ai = []
    t1b = []
    t1br = []
    t1bi = []
    t2a = []
    t2ar = []
    t2ai = []
    t2b = []
    t2br = []
    t2bi = []

    for m in srm:
        Em = []
        Emi = []
        for i in range(len(k)):
            R = m - D * (k[i] ** 2)
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
                    t1ar.append(Em)
                    t1ai.append(Emi)
                elif numpositivelocalmaxima > 0 and numZeroCrossing == 1:
                    t1b.append(m)
                    t1br.append(Em)
                    t1bi.append(Emi)
                elif numpositivelocalmaxima == 0 and numZeroCrossing % 2 == 1:
                    t2a.append(m)
                    t2ar.append(Em)
                    t2ai.append(Emi)
                elif numpositivelocalmaxima > 0 and numZeroCrossing % 2 == 1:
                    t2b.append(m)
                    t2br.append(Em)
                    t2bi.append(Emi)
    percent = (len(t1a) + len(t1b)) * 0.1
    return percent

def main():
    step_n = 1
    n = np.arange(3, 50 + step_n, step_n).tolist()
    step_var = 0.025
    var = np.arange(0, 1 + step_var, step_var).tolist()

    dp_list = [(x, y) for x in n for y in var]
    start_time = time.time()

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    # Monitor progress
    results = []
    for i, result in enumerate(pool.imap(process_tuple, dp_list), 1):
        results.append(result)
        if i % 100 == 0:
            print(f'Processed {i}/{len(dp_list)} tasks...')

    pool.close()
    pool.join()

    end_time = time.time()

    df_data = pd.DataFrame({'n': [x[0] for x in dp_list], 'var': [x[1] for x in dp_list], 'Percentage': results})
    df_data.to_csv(os.path.join('./', 'heatmap_fig7.csv'), index=False)
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    path = './heatmap_fig7.csv'
    data = pd.read_csv(path)

    n.insert(0, 2)  # insert n = 2 with % = 0 because Turing instability can't be computed for this case

    step = len(var)
    percentage = np.array(data['Percentage'])  # group Turing % values by network size n
    per = [percentage[i:i + step] for i in range(0, len(percentage), step)]
    per.insert(0, np.array([0] * step))

    Y, X = np.meshgrid(var, n)
    plt.figure(figsize=(8, 6))
    colormap = plt.pcolormesh(X, Y, per, cmap='viridis')
    plt.colorbar(colormap)
    plt.xlabel(r'n', fontsize=14)
    plt.ylabel('\u03C3\u00b2', fontsize=14)
    plt.title('s = 0%', fontsize=16, fontweight='bold')
    plt.savefig('heatmap_fig7.png')  # Save the figure as a PNG file
    plt.show()

    total_percentage = sum(percentage)
    share = []
    for i in per:
        k = (sum(i) / total_percentage) * 100
        share.append(k)
    plot = sns.lineplot(x=n, y=share)
    plot.set_xlabel('n', fontsize=16)
    plot.set_ylabel('%', fontsize=16)
    plot.set_title('Percentage by Network Size', fontsize=14, fontweight='bold')
    plot.set_xlim(n[0])
    plot.get_figure().savefig('percentage_by_network_size_fig7.png')  # Save the line plot as a PNG file
    plt.close()

if __name__ == '__main__':
    main()
