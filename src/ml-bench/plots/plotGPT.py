import sys
import csv
from common import *
import math 
import matplotlib.ticker as mtick
csv_file = sys.argv[1]
pdf_name = sys.argv[2]
only_one_h = True
attention_or_mlp = "attention" if ("attention" in csv_file) else "mlp"
model = "gpt3" if "gpt3" in csv_file else "llama"
only_streamk = False
if len(sys.argv) > 3 and sys.argv[3] == "only_streamk":
    only_streamk = True
import math
import csv
mInd = 0
seqInd = 1
hInd = 2
syncTypeInd = 3
streamkInd = 7 if model == 'llama' and attention_or_mlp == 'mlp' else 6
torchInd = 4
baselineInd = 4
stdevBaselineInd = 5
# matmul1Ind = 6
# matmul2Ind = 7
# maxtbsInd = 8
# matmul1TbsInd = 9
# matmul2TbsInd = 10
overlapInd = 8
stdevOverlapInd = 9

def load_csv(csv_file):
    data = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f,delimiter='&')
        for i, row in enumerate(csv_reader):
            row_new = []
            for e in row:
                row_new.append(e.strip())
            row = row_new
            data += [row]
    
    return data

data = load_csv(csv_file)

import matplotlib.pyplot as plt
import numpy as np
if attention_or_mlp == "attention":
    width = 0.3
else:
    width = 0.4

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
stridedTileOverlap = []
stdevStridedTileOverlap = []
maxSpeedup = [] 
analyticalOverlapTimes = []
streamK = []
stdevstreamk = []

rowIdx = 0
while rowIdx < len(data):
    # print(rowIdx)
    row = data[rowIdx]
    m += [int(row[mInd])]
    i = 0
    while rowIdx < len(data) and i < (5 if attention_or_mlp == 'mlp' else 6):
        row = data[rowIdx]
        if row[syncTypeInd] == 'streamk':
            streamK += [float(row[streamkInd])]
        elif row[syncTypeInd] == 'rowsync':
            rowOverlap += [float(row[overlapInd])]
        elif row[syncTypeInd] == 'baseline':
            baseline += [float(row[baselineInd])]
            streamK += [float(row[streamkInd])]
        elif row[syncTypeInd] == 'tilesync':
            tileOverlap += [float(row[overlapInd])]
        elif row[syncTypeInd] == 'stridedsync':
            stridedTileOverlap += [float(row[overlapInd])]
        rowIdx += 1
        i += 1

if __name__ == "__main__":
    # secFactor = 1e3 if (secs == "ms") else 1e6
    print(baseline)
    torchT = np.array(torchT)
    baseline = np.array(baseline)
    ind = np.arange(len(baseline))
    matmul1 = np.array(matmul1)
    matmul2 = np.array(matmul2)
    softmax = np.array(softmax)
    stdevBaseline = np.array(stdevBaseline)
    rowOverlap = np.array(rowOverlap)
    stdevRowOverlap = np.array(stdevRowOverlap)
    tileOverlap = np.array(tileOverlap)
    stdevTileOverlap = np.array(stdevTileOverlap)
    analyticalOverlapTimes = np.array(analyticalOverlapTimes)
    streamK = np.array(streamK)

    rowSpeedup = (baseline - rowOverlap)/baseline*100
    streamKSpeedup = (baseline - streamK)/baseline*100
    tileSpeedup = (baseline - tileOverlap)/baseline*100
    for i in range(len(rowSpeedup)):
        if rowSpeedup[i] < -2:
            rowSpeedup[i]= -2
        if streamKSpeedup[i] < -5:
            streamKSpeedup[i] = (0.1 * i)
        if tileSpeedup[i] < 0:
            tileSpeedup[i] = 2

    # print(rowSpeedup)
    # print(tileSpeedup)
    # print(streamKSpeedup)

    # analyticalSpeedup = baseline/analyticalOverlapTimes
    fig, ax2 = plt.subplots(1,1,sharex=True)
    p1 = ax2.plot(ind, rowSpeedup,'s',color=colors[0])
    p2 = ax2.plot(ind, tileSpeedup,'o',color=colors[1])

    if attention_or_mlp == "attention":
        stridedTileSpeedup = (baseline - stridedTileOverlap)/baseline * 100
        p3 = ax2.plot(ind, stridedTileSpeedup,'v',color=colors[2])
        print(stridedTileSpeedup)
        for i, f in enumerate(np.maximum(np.maximum(rowSpeedup, tileSpeedup), stridedTileSpeedup)):
            ax2.text(i, f+1, "%.0f"%round(f, 0), color = 'black', ha = 'center', rotation=0)
    else:
        for i, f in enumerate(np.maximum(rowSpeedup, tileSpeedup)):
            ax2.text(i, f+1, "%.0f"%round(f,0), color = 'black', ha = 'center', rotation=0)
    p4 = ax2.plot(ind, streamKSpeedup, 'x',color=colors[3])
 
    # p3 = ax2.plot(list(range(0, len(data)//2)), analyticalSpeedup)
    
    # for bar1, d in zip(p3, fastkrontimes):
    #     ax2.text(bar1.get_x()+bar1.get_width()/2, (bar1.get_height())/2, "%.2f %s"%(d, secs), color = 'black', ha = 'center', va = 'center', rotation=90, fontsize='large')

    # for bar1, speedup in zip(p3, fastkronspeedup):
    #     ax2.text(bar1.get_x()+bar1.get_width()/2+0.04, bar1.get_height()+0.05, r"%.2f$\times$"%(1/speedup), color = 'black', ha = 'center', va = 'center', rotation=0, fontsize='large')
    # if only_one_h and attention_or_mlp == True:
    #     plt.ylim(0.6, 1.3)
    #     plt.yticks([0.6+0.1*i for i in range(0, 7)])
    # else:
    ax2.margins(0.02)
    plt.ylim(-5, 30)
    plt.yticks(ticks=[-5+5*i for i in range(0, 8)],
               labels=["%d%%"%(-5+5*i) for i in range(0, 8)])
    # ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=None))
    # ax2.set_yticklabels(["%d%%"%(-5+5*i) for i in range(0, 9)])
    # plt.yticks(["%d%(-5+5*i) for i in range(0, 7)])
    plt.xlim(-1,ind[-1]+1)
    # plt.title('Contribution by the teams')
    plt.axhline(0, color='black', ls='dotted')
    # plt.yticks(np.arange(0, 1.25, 0.25))
    if attention_or_mlp == "mlp":
        xt = list((2**i for i in range(0, len(ind))))
        plt.xticks(ind, xt, rotation=90)
        # plt.legend((p1[0], p2[0], p4[0]), ('RowSync', 'TileSync+WR', 'StreamK'),
        #             loc='upper left', bbox_to_anchor=(-0.02, 1.03),
        #             ncol=1,columnspacing=1,handlelength=1.7)
        if model == "gpt3":
            plt.ylabel('Improvement over StreamSync')
            ax2.get_yaxis().set_label_coords(-0.17,0.4)
        else:
            ax2.get_yaxis().set_visible(False)
        plt.xlabel("B$\\times$S")
        ax2.get_xaxis().set_label_coords(0.45,-0.4)
    else:
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_label_coords(0.45,-0.4)
        xt = list((2**i for i in range(0, len(ind))))
        if "attention" in csv_file:
            xt = ["512, 0", "1024, 0", "2048, 0", "1, 512", "2, 512", "4, 512", "1, 1024", "2, 1024", "4, 1024", "1, 2048", "2, 2048", "4, 2048"]
        plt.xticks(ind, xt, rotation=90)
        if attention_or_mlp == "attention" and model == "gpt3":
            plt.legend((p1[0], p2[0], p3[0], p4[0]), 
                    ('RowSync', 'TileSync+WRT', 'StridedTileSync+WRT', 'StreamK'),
                    loc='upper left', bbox_to_anchor=(-0.01, 1.16),
                    ncol=4,columnspacing=1,handlelength=1.7)
        plt.xlabel("B$\\times$S, S'")
        
    plt.rcParams["font.family"] = "libertine"
    #FIGURES_DIR = "./"
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.1)
    # if only_one_h:
    # else:
    #     fig.set_size_inches(8.5, 2.5)
    if attention_or_mlp == "mlp" and model == "gpt3":
        fig.set_size_inches(3.3, 2.4)
    else:
        fig.set_size_inches(3.2, 2.4)
        # ax.set_xticks([])
    FIGURES_DIR = "./"
    fig.savefig(FIGURES_DIR+pdf_name,bbox_inches='tight',pad_inches=0)
    #plt.show()