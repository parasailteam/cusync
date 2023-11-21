import sys
import csv
from common import *
import math 

mlp_csv = sys.argv[1]
attention_csv = sys.argv[2]
# allreduce_csv = sys.argv[3]
pdf_name = sys.argv[3]

def load_data(csv_file):
    mInd = 0 
    hInd = 1
    syncTypeInd = 2
    torchInd = 3
    baselineInd = 4
    stdevBaselineInd = 5
    matmul1Ind = 6
    matmul2Ind = 7
    maxtbsInd = 8
    matmul1TbsInd = 9
    matmul2TbsInd = 10
    overlapInd = 11
    stdevOverlapInd = 12

    def load_csv(csv_file):
        data = []
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f,delimiter='&')
            header = next(csv_reader)
            for i, row in enumerate(csv_reader):
                row_new = []
                for e in row:
                    row_new.append(e.strip())
                row = row_new
                if (int(row[hInd]) != 12288):
                    continue
                if True or attention_or_mlp == False:
                    if (math.log(int(row[mInd]), 2).is_integer() or int(row[mInd]) == 2048): #int(row[hInd]) == hidden and 
                        data += [row]
                else:
                    data += [row]
        
        return data

    data = load_csv(csv_file)
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
    stridedTileSyncOverlap = []
    stdevStridedTileSyncOverlap = []
    maxSpeedup = []
    analyticalOverlapTimes = []
    streamK = []
    rowIdx = 0
    while rowIdx < len(data):
        print(rowIdx)
        row = data[rowIdx]
        print(row)
        m += [int(row[mInd])]
        h += [int(row[hInd])]
        torchT += [float(row[torchInd])]
        baseline += [float(row[baselineInd])]
        stdevBaseline += [float(row[stdevBaselineInd])]
        matmul1 += [float(row[matmul1Ind])]
        matmul2 += [float(row[matmul2Ind])]
        matmul1Tbs = int(row[matmul1TbsInd])
        matmul2Tbs = int(row[matmul2TbsInd])
        matmul1Time = float(row[matmul1Ind])
        matmul2Time = float(row[matmul2Ind])
        # if not (int(row[mInd]) < 1024 or matmul1Tbs < 2*80):
        #     MaxTBs = 2*80
        #     waves1 = (matmul1Tbs + MaxTBs - 1)//MaxTBs
        #     lastWave1TBs = matmul1Tbs % MaxTBs
        #     timePerFullWave1 = matmul1Time/(waves1 - 1 + lastWave1TBs/MaxTBs)
        #     timePerPartialWave1 = timePerFullWave1 * lastWave1TBs/MaxTBs
            
        #     lastWave2TBs = matmul2Tbs % MaxTBs
        #     waves2 = (matmul2Tbs + MaxTBs - 1)//MaxTBs
        #     timePerFullWave2 = matmul2Time/(waves2 - 1 + lastWave2TBs/MaxTBs)

        #     overlapWaves = (matmul2Tbs + matmul1Tbs + MaxTBs - 1)//MaxTBs
        #     firstWave2TBs = MaxTBs - lastWave1TBs
        #     remainingWaves2 = (matmul2Tbs - firstWave2TBs)//MaxTBs
        #     timeFirstOverlappedWave2 = (firstWave2TBs/MaxTBs) * timePerFullWave2
        #     remainingOverlappedTBs2 = matmul2Tbs - remainingWaves2 * MaxTBs
        #     analyticalOverlapTime = (waves1 - 1)*timePerFullWave1 + max(timePerPartialWave1, timeFirstOverlappedWave2) + (remainingWaves2-1)*timePerFullWave2 + remainingOverlappedTBs2/MaxTBs * timePerFullWave2
        #     analyticalOverlapTimes += [analyticalOverlapTime]
        #     print(row[overlapInd], analyticalOverlapTime, timePerFullWave1, timePerFullWave2, timePerPartialWave1)
        # else:
        #     analyticalOverlapTimes += [matmul1Time + matmul2Time]
        # softmax += [float(row[softmaxInd])]
        rowOverlap += [float(row[overlapInd])]
        stdevRowOverlap += [float(row[stdevOverlapInd])]
        rowIdx += 1
        row = data[rowIdx]
        tileOverlap += [float(row[overlapInd])]
        stdevTileOverlap += [float(row[stdevOverlapInd])]
        rowIdx += 1
        row = data[rowIdx]
        streamK += [float(row[overlapInd])]
        rowIdx += 1
        if 'attention' in csv_file:
            row = data[rowIdx]
            stridedTileSyncOverlap += [float(row[overlapInd])]
            stdevStridedTileSyncOverlap += [float(row[stdevOverlapInd])]
            rowIdx += 1

    if 'attention' in csv_file:
        return (baseline, rowOverlap, tileOverlap, stridedTileSyncOverlap, streamK)
    else:
        return (baseline, rowOverlap, tileOverlap, streamK)

import matplotlib.pyplot as plt
import numpy as np
# ind = np.arange(len(data)/2)
# if not only_one_h:
#     for i in range(len(ind)):
#         ind[i] += i//(len(ind)/3)
width = 0.44
# fig = plt.subplots(figsize =(10, 7))
attention_results = load_data(attention_csv)
mlp_results = load_data(mlp_csv)
ind = np.arange(len(attention_results[0]))
print(ind)
fig, ax2 = plt.subplots(1,1,sharex=True)
# print(matmul1)
# print(overlap)
# print(matmul2)

# p1 = ax2.bar(ind, matmul1, width, align = 'edge',color=colors[0])
# p2 = ax2.bar(ind, softmax, width, bottom = matmul1,align='edge',color=colors[1])
# p3 = ax2.bar(ind, matmul2, width, bottom = softmax+matmul1,align='edge',color=colors[2])
attention_speedup = attention_results[4]/np.minimum(np.minimum(attention_results[1], attention_results[2]), attention_results[3])
mlp_speedup = mlp_results[3]/np.minimum(mlp_results[1], mlp_results[2])
print(mlp_results[0], mlp_results[1], mlp_results[2], mlp_results[3])

print(attention_speedup)
print(mlp_speedup)
p1 = ax2.bar(ind, attention_speedup, width, align='edge',color=colors[0])
p2 = ax2.bar(ind + 0.01+ width, mlp_speedup, width, align='edge',color=colors[1])

d = 0
for bar1, bar2 in zip(p1.patches, p2.patches):
    ax2.text(bar1.get_x()+bar1.get_width()/2, bar1.get_height() + 0.1, "%.2f"%(attention_speedup[d]), color = 'black', ha = 'center', va = 'center', rotation=90)
    ax2.text(bar2.get_x()+bar2.get_width()/2, bar2.get_height() + 0.1, "%.2f"%(mlp_speedup[d]), color = 'black', ha = 'center', va = 'center', rotation=90)
    d += 1

plt.ylim(0.6, 1.5)
plt.yticks([0.6+0.1*i for i in range(0, 10)])
plt.ylabel('Speedup over StreamK', fontsize='large')
plt.xlabel("Batch Size", fontsize='large')
xt = [1,2,4,8,16,32,64,128,256,512,1024,2048]
plt.xticks(ind+width, xt, rotation=90)
plt.axhline(1.0, color='black', ls='dotted')
plt.legend((p1[0], p2[0]), ('Self-Attention', 'MLP'),
            loc='upper left', bbox_to_anchor=(-0.01, 1.02),
            ncol=4,columnspacing=1,handlelength=1.7)
FIGURES_DIR = "./"
    
plt.rcParams["font.family"] = "libertine"
#FIGURES_DIR = "./"
fig = plt.gcf()
fig.subplots_adjust(bottom=0.1)
fig.set_size_inches(4.7, 2.3)
# ax.set_xticks([])
fig.savefig(FIGURES_DIR+pdf_name,bbox_inches='tight',pad_inches=0)
# plt.show()