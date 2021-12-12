from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import scipy.stats as ss
import seaborn
from pingouin import anova


def task_1():
    X_ms = [24.80, 24.84, 24.58, 21.73, 24.79, 21.00, 26.57, 18.31, 25.01, 21.71, 22.53, 19.25, 23.67, 21.17, 25.10,
            21.82, 26.71, 23.14, 22.31, 21.32, 23.83, 19.18, 22.79, 20.25, 24.97, 22.57, 24.64, 21.12, 26.04, 21.34,
            22.79, 21.76, 25.49, 19.82, 25.37, 20.30, 26.76, 20.27, 25.32, 22.67, 25.31, 20.93, 26.19, 19.81, 25.06,
            20.63, 23.95, 22.10, 24.91, 21.41]
    Y_ms = [28.08, 21.40, 25.90, 25.24, 23.18, 24.87, 27.56, 24.03, 25.96, 25.37, 21.85, 22.15, 24.88, 21.75, 23.01,
            23.99, 22.05, 25.92, 24.45, 23.05, 24.07, 23.11, 24.14, 26.93, 20.31, 20.95, 23.71, 25.07, 23.76, 24.92,
            23.70, 22.14, 23.70, 25.61, 25.27, 24.38, 22.57, 24.56, 23.69, 22.79, 22.95, 27.71, 26.20, 25.05, 26.20,
            24.41, 21.95, 22.56, 20.31, 23.43]

    seaborn.histplot(X_ms, binwidth=0.75)
    data = np.random.normal(23, 2.25, 100)
    mu, std = ss.norm.fit(data)
    x = np.linspace(16, 30, 100)
    p = ss.norm.pdf(x, mu, std)
    plt.plot(x, p * 45, 'k', linewidth=2)
    plt.show()

    seaborn.histplot(Y_ms, binwidth=0.75)
    data = np.random.normal(24, 2.4, 100)
    mu, std = ss.norm.fit(data)
    x = np.linspace(17, 31, 100)
    p = ss.norm.pdf(x, mu, std)
    plt.plot(x, p * 50, 'k', linewidth=2)
    plt.show()

    print(f"X: {ss.normaltest(X_ms)}")  # NOT normal
    print(f'X: {ss.shapiro(X_ms)}')  # IS normal
    print(f'X: {ss.anderson(X_ms)}', )  # NOT normal
    print(f"Y: {ss.normaltest(Y_ms)}")  # IS normal
    print(f'Y: {ss.shapiro(Y_ms)}')  # IS normal
    print(f'Y: {ss.anderson(Y_ms)}', )  # IS normal

    print(ss.mannwhitneyu(X_ms, Y_ms))
    Ux = ss.mannwhitneyu(X_ms, Y_ms).statistic
    m = 50
    n = 50
    asymptotic_MW_test = (Ux - m * n / 2) / np.sqrt(m * n * (m + n + 1) / 12)
    print(f"Testovací kritérium u = {asymptotic_MW_test}")

    print(ss.mannwhitneyu(X_ms, Y_ms, alternative="less"))  # F(u) > G(u)
    print(ss.mannwhitneyu(X_ms, Y_ms, alternative="greater"))  # F(u) < G(u)
    print(ss.kruskal(X_ms, Y_ms))
    print(f"co-W = {format(ss.chi2.ppf(0.95, 1), '.4f')}")


def task_2():
    f1_names = ["rano", "poledne", "vecer"]
    f2_names = ["ticho", "hudba", "hluk", "krik"]

    data = [
        [[6, 8, 11, 7], [7, 8, 12, 10], [8, 7, 20], [13, 21]],
        [[8, 13, 7, 6], [5, 11, 7], [10, 17, 11, 13], [14]],
        [[7, 8, 6], [6, 8, 16, 15], [12, 17, 8], [13, 17, 15, 22, 18]]
    ]

    records = []
    combinations = list(product(enumerate(f1_names), enumerate(f2_names)))
    combinations = [(a, b, c, d) for (a, b), (c, d) in combinations]
    for i, f1, j, f2 in combinations:
        [records.append([f1, f2, n]) for n in data[i][j]]

    column_names = ["f1", "f2", "time"]

    df = pandas.DataFrame(records, columns=column_names)

    ret = anova(data=df, dv="time", between=["f1", "f2"], ss_type=2, detailed=True, effsize="n2")
    ret["co-W_right"] = [3.3404, 2.9467, 2.4453, np.nan]
    ret = ret[["Source", "DF", "F", "co-W_right", "p-unc"]]
    print(ret)


def categorical_analysis(data, columns=None):
    method = "pearson"
    alpha = 0.05

    chi2, p_value, dof, theo_freq = ss.chi2_contingency(data, lambda_=method)
    chi2_co_W = format(ss.chi2.ppf(1 - alpha, (len(data) - 1) * (len(data[0]) - 1)), '.4f')
    print(
        f"\nX^2 = {format(chi2, '.4f')} | p-value = {format(p_value, '.4f')}"
        f" | dof = {dof}, X^2(dof={dof}) = {chi2_co_W}")
    theo_freq = pd.DataFrame(theo_freq, columns=columns)
    pd.set_option('display.float_format', lambda x: '%0.2f' % x)
    print(f"{theo_freq}")


def task_3():
    data_orig = [[9, 22, 27, 14, 0], [6, 10, 18, 10, 0], [0, 0, 0, 2, 0],
                 [2, 2, 3, 1, 0], [0, 2, 3, 1, 0], [1, 1, 3, 2, 1]]
    data_red_1 = [[9, 22, 27, 14], [6, 10, 18, 12], [3, 5, 9, 5]]
    data_red_2_lepsi = [[31, 27, 14], [16, 18, 12], [8, 9, 5], ]
    data_red_2_horsi = [[9, 22, 41], [6, 10, 30], [3, 5, 14]]
    data_red_3 = [[31, 41], [16, 30], [8, 14]]

    categorical_analysis(data_red_1, columns=["1.0-1.4", "1.5-1.9", "2.0-2.4", "2.5 a vice"])
    categorical_analysis(data_red_2_lepsi, columns=["1.0-1.9", "2.0-2.4", "2.5 a vice"])
    categorical_analysis(data_red_2_horsi, columns=["1.0-1.4", "1.5-1.9", "2.0 a vice"])
    categorical_analysis(data_red_3, columns=["1.0-1.9", "2.0 a vice"])


if __name__ == '__main__':
    task_2()
