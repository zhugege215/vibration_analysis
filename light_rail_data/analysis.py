#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

mpl.rcParams['agg.path.chunksize'] = 10000
pd.set_option('display.width', 200)
dateparse = lambda x: pd.datetime.fromtimestamp(float(x))
names = ['time', 'data']


class lightRailData_analysis:
    'the process&analysis of light rail data'

    def __init__(self, filename):
        self.originalFilename = filename
        self.processFilename = filename.split('.')[0].split('\\')[4] + "_process.csv"
        self.timeProcessFilename = filename.split('.')[0].split('\\')[4] + "_process_time.csv"
        self.timeProcessFilenamePro = filename.split('.')[0].split('\\')[4] + "_process_time_pro.csv"

    def txt_process(self):
        f1 = open(self.processFilename, 'w')
        with open(self.originalFilename) as f:
            for i, line in enumerate(f):
                if i >= 17:
                    l = re.sub(r'\s{3,}', '', line, count=1)
                    f1.write(re.sub(r'\s{3,}', ',', l))  # 这里的空格不对，我也是醉了
        f1.close()

    def time_process(self):
        df = pd.read_csv(self.processFilename, names=names, header=None)
        df['time'] = df['time'] + 1509976830
        df['data'] = df['data'] / 50.0  # 单位转换： 单位 g （10m/s2）
        df.to_csv(self.timeProcessFilename, index=None, header=None, float_format='%.6f', mode='w')

    def data_process(self, delta):
        f = open(self.timeProcessFilename)
        index = 0
        a = float(f.readline().strip().split(',')[1])
        while 1:
            b = float(f.readline().strip().split(',')[1])
            index += 1
            if abs(b - a) > delta:
                break
            a = b
        f.close()

        f1 = open(self.timeProcessFilenamePro, 'w')
        with open(self.timeProcessFilename) as f:
            for i, line in enumerate(f):
                if i >= index:
                    f1.write(line)
        f1.close()

    def show(self, name):
        if name == "process":
            filename = self.processFilename
        elif name == "time":
            filename = self.timeProcessFilename
        elif name == "pro":
            filename = self.timeProcessFilenamePro

        # df_10 = pd.read_csv(filename, names=names, nrows=10, parse_dates=['time'])
        # print df_10
        df = pd.read_csv(filename, names=names, parse_dates=['time'], date_parser=dateparse, index_col='time')
        df["data"].plot()
        plt.show()

    def fourier(self, Fs, start, end):
        df = pd.read_csv(self.timeProcessFilenamePro, names=['time', 'data'], parse_dates=['time'],
                         date_parser=dateparse, index_col='time')
        df_p = df[start:end]
        x = df_p['data'].values
        y = df_p['data'].values
        n = len(x)  # n表示采样点数
        print n
        k = np.arange(n)  # 构造长度为n的数组
        print k
        # t = n / Fs  # T 表示采样时长 delt_t
        # frq = k / t # 构造傅里叶变换的横坐标
        frq = k / float(n) * Fs
        print frq
        # exit(0)
        frq2 = frq[range(int(n / 2))]

        yf = abs(np.fft.fft(y))  # 做傅里叶变换
        yf1 = abs(np.fft.fft(y)) / n
        yf2 = yf1[range(int(len(x) / 2))] * 2

        plt.subplot(221)
        # plt.plot(x[0:500], y[0:500])
        df_p['data'].plot()
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
    filename = ur"G:\新联铁\地铁数据\地铁数据第二天_恶劣\21.txt"  # G:\新联铁\地铁数据\地铁数据第二天_恶劣  G:\新联铁\地铁数据\地铁数据\12.txt
    ob = lightRailData_analysis(filename)
    # print ob.__doc__
    # ob.txt_process()
    # ob.time_process()
    # ob.data_process(48) # 2000,48
    # ob.fourier(5120, '2017-11-06 22:20:15', '2017-11-06 22:24:27')
    ob.show("time")
    # x = [1,2,3]
    # y = [1,2,3]
    # plt.plot(x, y)
    # plt.show()
