#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
import matplotlib.pyplot as plt

# 数据预处理
def txt_process():
    f1 = open('2_process.csv', 'w')
    with open("2.txt") as f:
        for i, line in enumerate(f):
            if i >= 15:
                l = line.replace("          ", "",1)
                f1.write(l.replace("          ", ","))
    f1.close()

# 数据清洗
def data_process(filename, delta):
    f = open(filename)
    index = 0
    a = float(f.readline().strip().split(',')[1])
    while 1:
        b = float(f.readline().strip().split(',')[1])
        index += 1
        if abs(b-a) > delta:
            break
        a = b
    f.close()

    f1 = open(filename.split('.')[0] + '_pro.csv', 'w')
    with open(filename) as f:
        for i, line in enumerate(f):
            if i >= index:
                f1.write(line)
    f1.close()

# 简单绘制并查看数据
def show(filename):
    names = ['time', 'data']
    df_10 = pd.read_csv(filename, names=names, nrows=10, parse_dates=['time'])
    print df_10
    df = pd.read_csv(filename, names=names, parse_dates=['time'], index_col='time')
    df["data"].plot()
    plt.show()

# 对信号进行傅里叶变换
# 传入的Fs主要是使最终画出的fft的横坐标正确
def fourier(filename, Fs):
    df = pd.read_csv(filename, names = ['time', 'data'])
    x = df['time'].values
    y = df['data'].values
    n = len(x)  # n表示采样点数
    k = np.arange(n)    # 构造长度为n的数组
    # t = n / Fs  # T 表示采样时长 delt_t
    # frq = k / t # 构造傅里叶变换的横坐标
    frq = k * Fs / n
    print frq
    frq2 = frq[range(int(n / 2))]

    yf = abs(np.fft.fft(y)) # 做傅里叶变换
    yf1 = abs(np.fft.fft(y)) / n
    yf2 = yf1[range(int(len(x) / 2))] * 2

    plt.subplot(221)
    # plt.plot(x[0:500], y[0:500])
    plt.plot(x,y)
    plt.title('Original wave')

    plt.subplot(222)
    plt.plot(frq, yf, 'r')
    plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')

    plt.subplot(223)
    plt.plot(frq, yf1, 'g')
    plt.title('FFT of Mixed wave(normalization)', fontsize=9, color='r')

    plt.subplot(224)
    plt.plot(frq2, yf2, 'b')
    plt.title('FFT of Mixed wave)', fontsize=10, color='#F08080')

    plt.show()

if __name__ == "__main__":
    # txt_process()
    # data_process('1_process.csv',1000)
    # show('1_process_pro.csv')
    fourier('1_process_pro.csv', 2048)