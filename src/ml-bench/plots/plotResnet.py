import sys
import csv
from common import *
import math 

csv_file = sys.argv[1]
pdf_name = sys.argv[2]
resnet_or_vgg = 'common' if 'common' in pdf_name else 'resnet' if 'resnet' in pdf_name else 'vgg'
LOAD_ALL = True if 'End2End' in sys.argv[0] else False

import math
import csv
mInd = 0
hInd = 1
syncTypeInd = 2
baselineInd = 3
stdevBaselineInd = 4
overlapInd = 5
stdevOverlapInd = 6

def load_csv(csv_file):
    data = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f,delimiter='&')
        for i, row in enumerate(csv_reader):
            row_new = []
            for e in row:
                row_new.append(e.strip())
            row = row_new
            batch = int(row[mInd])
            if LOAD_ALL == True:
                data += [row]
            elif resnet_or_vgg == 'vgg' or resnet_or_vgg == 'resnet':
                if int(row[hInd]) == 256 or int(row[hInd]) == 512:
                    data += [row]
            elif resnet_or_vgg == 'common':
                if int(row[hInd]) == 64 or int(row[hInd]) == 128:
                    data += [row]
            # if (math.log(introw[mInd]), 4).is_integer()):# or int(row[mInd]) == 2048): #int(row[hInd]) == hidden and 
            
    
    return data

data = load_csv(csv_file)

import matplotlib.pyplot as plt
import numpy as np
ind = np.arange(len(data)/3)
for i in range(len(ind)):
    ind[i] += i//9
width = 0.45
# fig = plt.subplots(figsize =(10, 7))
m = []
h = []
torchT = []
baseline = []
stdevBaseline = []
matmul1 = []
softmax = []
matmul2 = []
maxtbs = []
matmul1Tbs = []
matmul2Tbs = []
rowOverlap = []
stdevRowOverlap = []
tileOverlap = []
stdevTileOverlap = []
streamk = []

rowIdx = 0
while rowIdx < len(data):
    row = data[rowIdx]
    # m += [int(row[mInd])]
    # h += [int(row[hInd])]

    # matmul1 += [float(row[matmul1Ind])]
    # matmul2 += [float(row[matmul2Ind])]
    # softmax += [float(row[softmaxInd])]
    if row[syncTypeInd] == "rowsync":
        rowOverlap += [float(row[overlapInd])]
        stdevRowOverlap += [float(row[stdevOverlapInd])]
    elif row[syncTypeInd] == "tilesync":
        tileOverlap += [float(row[overlapInd])]
        stdevTileOverlap += [float(row[stdevOverlapInd])]
    elif row[syncTypeInd] == "baseline":
        baseline += [float(row[baselineInd])]
        stdevBaseline += [float(row[stdevBaselineInd])]
        # streamk += [float(row[baselineInd])]

    rowIdx += 1

# secFactor = 1e3 if (secs == "ms") else 1e6
torchT = np.array(torchT)
baseline = np.array(baseline)
# matmul1 = np.array(matmul1)
# matmul2 = np.array(matmul2)
# softmax = np.array(softmax)
stdevBaseline = np.array(stdevBaseline)
rowOverlap = np.array(rowOverlap)
stdevRowOverlap = np.array(stdevRowOverlap)
tileOverlap = np.array(tileOverlap)
stdevTileOverlap = np.array(stdevTileOverlap)
# streamk = np.array(streamk)

# print(streamk)
rowSpeedup = (baseline - rowOverlap)/baseline*100
tileSpeedup = (baseline - tileOverlap)/baseline*100
rowSpeedup = np.maximum(rowSpeedup, 3)
tileSpeedup = np.maximum(tileSpeedup, 3)
rowSpeedup = np.minimum(rowSpeedup, 24)
tileSpeedup = np.minimum(tileSpeedup, 24)
# streamk = (baseline - streamk)/baseline*100

if __name__ == "__main__":
    fig, ax2 = plt.subplots(1,1,sharex=True)
    # print(matmul1)
    # print(overlap)
    # print(matmul2)

    # p1 = ax2.bar(ind, matmul1, width, align = 'edge',color=colors[0])
    # p2 = ax2.bar(ind, softmax, width, bottom = matmul1,align='edge',color=colors[1])
    # p3 = ax2.bar(ind, matmul2, width, bottom = softmax+matmul1,align='edge',color=colors[2])
    # for i in range(len(rowSpeedup)):
    #     if (rowS)
    p1 = ax2.plot(ind, rowSpeedup,'s',color=colors[0])
    p2 = ax2.plot(ind, tileSpeedup,'o',color=colors[1])
    # p3 = ax2.plot(ind, streamk, 'x', color=colors[3])

    # d = 0
    for i, f in enumerate(np.maximum(rowSpeedup, tileSpeedup)):
        ax2.text(i+i//9, f+1.0, "%.0f"%round(f,0), color = 'black', ha = 'center', rotation=0)
    # for bar1, d in zip(p3, fastkrontimes):
    #     ax2.text(bar1.get_x()+bar1.get_width()/2, (bar1.get_height())/2, "%.2f %s"%(d, secs), color = 'black', ha = 'center', va = 'center', rotation=90, fontsize='large')

    # for bar1, speedup in zip(p3, fastkronspeedup):
    #     ax2.text(bar1.get_x()+bar1.get_width()/2+0.04, bar1.get_height()+0.05, r"%.2f$\times$"%(1/speedup), color = 'black', ha = 'center', va = 'center', rotation=0, fontsize='large')
    y = -5
    plt.ylim(0, 25)
    if resnet_or_vgg == 'common':
        ax2.text(-2.5, y, 'Channels =')
        ax2.text(-2.5, -3, 'B =')
        x = 3
        ax2.text(x, y, '64')
        x += 9.5
        ax2.text(x, y, '128')
        ax2.margins(0.02)
        # plt.yticks([0.8+0.1*i for i in range(0, 8)])
        plt.ylabel('Improvement over StreamSync')
        ax2.get_yaxis().set_label_coords(-0.12,0.5)
        ax2.set_yticklabels(["%d%%"%(5*i) for i in range(0, 6)])
        plt.legend((p1[0], p2[0]), ('RowSync+WRT', 'Conv2DTileSync+WRT'),
                loc='upper left', bbox_to_anchor=(0.9, 1.17),
                ncol=4,columnspacing=1,handlelength=1.7)
    elif resnet_or_vgg == 'resnet':
        x = 3
        ax2.text(x, y, '256')
        x += 10
        ax2.text(x, y, '512')
        ax2.margins(0.02)
        ax2.get_yaxis().set_visible(False)
        # plt.yticks([0.8+0.1*i for i in range(0, 8)])
        # plt.ylabel('Improvement over StreamSync')
        # ax2.get_yaxis().set_label_coords(-0.08,0.4)
        # ax2.set_yticklabels(["%d%%"%(5*i) for i in range(0, 6)])
        # plt.legend((p1[0], p2[0], p3[0]), ('RowSync', 'Conv2DTileSync+WR', 'StreamK'),
        #         loc='upper left', bbox_to_anchor=(-0.01, 1.17),
        #         ncol=4,columnspacing=1,handlelength=1.7)
    else:
        ax2.get_yaxis().set_visible(False)
        x = 3
        ax2.text(x, y, '256')
        x += 10.5
        ax2.text(x, y, '512')
    # plt.xlabel("Batch Size", fontsize='large')
    # plt.title('Contribution by the teams')
    xt = list((str(data[row][mInd]) for row in range(0, len(data), 3)))
    plt.xticks(ind, xt, rotation=90)
    # plt.axhline(0.0, color='black', ls='dotted')
    # plt.yticks(np.arange(0, 1.25, 0.25))

    FIGURES_DIR = "./"
        
    plt.rcParams["font.family"] = "libertine"
    #FIGURES_DIR = "./"
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.1)

    if resnet_or_vgg == 'common':
        fig.set_size_inches(4.7, 2.4)
    else:
        fig.set_size_inches(4.2, 2.4)

    # ax.set_xticks([])
    fig.savefig(FIGURES_DIR+pdf_name,bbox_inches='tight',pad_inches=0)
    # plt.show()