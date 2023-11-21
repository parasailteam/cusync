import sys
import csv
from common import *
import math 

mlp_gpt3_csv = sys.argv[1]
attention_gpt3_csv = sys.argv[2]
mlp_llama_csv = sys.argv[3]
attention_llama_csv = sys.argv[4]

# allreduce_csv = sys.argv[3]
pdf_name = sys.argv[5]

def load_csv(csv_file):
    model = "gpt-3" if "gpt-3" in csv_file else "llama"
    attention_or_mlp = "attention" if ("attention" in csv_file) else "mlp"
    mInd = 0
    seqInd = 1
    hInd = 2
    syncTypeInd = 3
    streamkInd = 7 if model == 'llama' and attention_or_mlp == 'mlp' else 6
    torchInd = 4
    baselineInd = 4
    stdevBaselineInd = 5
    overlapInd = 8
    stdevOverlapInd = 9
    data = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f,delimiter='&')
        for i, row in enumerate(csv_reader):
            row_new = []
            for e in row:
                row_new.append(e.strip())
            row = row_new
            data += [row]
    
    import matplotlib.pyplot as plt
    import numpy as np

    # fig = plt.subplots(figsize =(10, 7))
    h = []
    torchT = []
    baseline = {}
    rowOverlap = {}
    tileOverlap = {}
    stridedTileOverlap = {}
    streamK = {}
    
    rowIdx = 0
    while rowIdx < len(data):
        row = data[rowIdx]
        m = int(row[mInd])
        seq = int(row[seqInd])
        if row[syncTypeInd] == 'streamk':
            streamK[(m, seq)] = float(row[streamkInd])
        elif row[syncTypeInd] == 'rowsync':
            rowOverlap[(m, seq)] = float(row[overlapInd])
        elif row[syncTypeInd] == 'tilesync':
            tileOverlap[(m,seq)] = float(row[overlapInd])
        elif row[syncTypeInd] == 'stridedsync':
            stridedTileOverlap[(m,seq)] = float(row[overlapInd])
        elif row[syncTypeInd] == 'baseline':
            baseline[(m,seq)] = float(row[baselineInd])
        rowIdx += 1
        
    if 'attention' in csv_file:
        return (baseline, rowOverlap, tileOverlap, stridedTileOverlap)
    else:
        return (baseline, rowOverlap, tileOverlap)

def allreduce_times(csv_file):
    data = {}
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f,delimiter='&')
        for i, row in enumerate(csv_reader):
            row_new = []
            for e in row:
                row_new.append(e.strip())
            row = row_new
            data[int(row[0])] = float(row[2]) 
    return data

cases = [(512, 0), (1024,0), (2048,0), (1, 512), (2, 512), (4, 512), (1, 1024), (2, 1024), (4, 1024), (1, 2048), (2, 2048), (4, 2048)]

def end2endTime(mlpTime, attentionTime, allreduceTime):
    e2e = []
    for case in cases:
        mlp = min([t[(case[0], 0)] for t in mlpTime])
        atn = min([t[case] for t in attentionTime])
        ar = allreduceTime[case[0]]
        e2e += [mlp+atn+2*ar]
    return np.array(e2e)

import matplotlib.pyplot as plt
import numpy as np
# ind = np.arange(len(data)/2)
# if not only_one_h:
#     for i in range(len(ind)):
#         ind[i] += i//(len(ind)/3)
width = 0.5

def end2EndResults(attention_csv, mlp_csv, allreduce_csv):
    attention_results = load_csv(attention_csv)
    mlp_results = load_csv(mlp_csv)
    ar = allreduce_times(allreduce_csv)
    
    end2endBaselineTimes = end2endTime([mlp_results[0]], [attention_results[0]], ar)    
    end2endOverlap = end2endTime(mlp_results[1:], attention_results[1:], ar)
    end2EndSpeedup = (end2endBaselineTimes-end2endOverlap)/end2endBaselineTimes*100
    return end2EndSpeedup

end2EndSpeedupGPT3 = end2EndResults(attention_gpt3_csv,mlp_gpt3_csv,"allreduce-gpt-3.csv")
end2EndSpeedupLLAMA = end2EndResults(attention_llama_csv,mlp_llama_csv,"allreduce-llama.csv")
print(end2EndSpeedupLLAMA)
print(end2EndSpeedupGPT3)

ind = np.arange(len(end2EndSpeedupGPT3))
fig, ax2 = plt.subplots(1,1,sharex=True)

p1 = ax2.plot(ind, end2EndSpeedupGPT3,'s',color=colors[0])
p2 = ax2.plot(ind, end2EndSpeedupLLAMA,'o',color=colors[1])

# d = 0
# for i, f in enumerate(end2EndSpeedupGPT3):
#     ax2.text(i, f, "%.1f"%round(f,1), color = 'black', ha = 'center')
#     d += 1
plt.legend((p1[0], p2[0]), ('GPT-3', 'LLaMA'),
           loc='upper left', bbox_to_anchor=(0.1, 1.19),
           ncol=2,columnspacing=1,handlelength=0)

plt.ylim(0, 24)
plt.yticks(ticks=[4*i for i in range(0, 7)], labels=["%d%%"%(4*i) for i in range(0, 7)])
for i, f in enumerate(zip(end2EndSpeedupGPT3, end2EndSpeedupLLAMA)):
    ax2.text(i, max(f)+1, "%.0f"%round(max(f),0), color = 'black', ha = 'center', rotation=0)
# ax2.set_yticklabels(["%d%%"%(2+2*i) for i in range(1, 10)])
plt.ylabel('Reduction in Inference Times')
ax2.get_yaxis().set_label_coords(-0.20,0.4)
xt = list(cases)
plt.xticks(ind, xt, rotation=90)
plt.xlabel("(B$\\times$S, S')")

FIGURES_DIR = "./"
    
plt.rcParams["font.family"] = "libertine"
#FIGURES_DIR = "./"
fig = plt.gcf()
fig.subplots_adjust(bottom=0.1)
fig.set_size_inches(2.9, 2.2)
# ax.set_xticks([])
fig.savefig(FIGURES_DIR+pdf_name,bbox_inches='tight',pad_inches=0)
# plt.show()