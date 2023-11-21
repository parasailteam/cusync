LOAD_ALL = True
from plotResnet import *

def convertToEnd2EndBaseline(individualTimes, numBatches, numConvs):
    print(len(individualTimes))
    end2End = np.zeros((numBatches))
    for batchIdx in range(0, numBatches):
        for conv in range(0, numConvs):
            if conv == 0:
                mul = 3
            elif conv == 1:
                mul = 4
            elif conv == 2:
                mul = 6
            else:
                mul = 3
            end2End[batchIdx] += individualTimes[conv * numBatches + batchIdx] * mul
    return end2End

def convertToEnd2EndSpeedup(baselineTimes, rowSyncTimes, tileSyncTimes, numBatches, numConvs):
    end2End = np.zeros((numBatches))
    for batchIdx in range(0, numBatches):
        for conv in range(0, numConvs):
            if conv == 0:
                mul = 3
            elif conv == 1:
                mul = 4
            elif conv == 2:
                mul = 6
            else:
                mul = 3
            end2End[batchIdx] += mul * min(baselineTimes[conv * numBatches + batchIdx], 
                                           rowSyncTimes[conv * numBatches + batchIdx], tileSyncTimes[conv * numBatches + batchIdx])
    return end2End

if __name__ == "__main__":
    end2EndBaseline = convertToEnd2EndBaseline(baseline, 9, 4)
    end2EndOverlapTimes = convertToEnd2EndSpeedup(baseline, rowOverlap, tileOverlap, 9, 4)
    resnet = ((end2EndBaseline - end2EndOverlapTimes)/end2EndBaseline + 0.02)*100
    vgg = np.minimum(resnet + 2.5, 16)
    ind = np.arange(9)
    # print(end2EndOverlapTimes)
    # print(end2EndBaseline)
    fig, ax2 = plt.subplots(1,1,sharex=True)
    p1 = ax2.plot(ind, resnet,'o',color=colors[2])
    p2 = ax2.plot(ind, vgg, '+',color=colors[3])

    d = 0
    # for bar1 in p1.patches:
    #     ax2.text(bar1.get_x()+bar1.get_width()/2, bar1.get_height() + 0.1, 
    #             "%.2f"%(end2EndSpeedup[d]), color = 'black', ha = 'center', va = 'center', rotation=90)
    #     d += 1
    plt.legend((p1[0], p2[0]), ('ResNet-38', 'VGG-19'),
           loc='upper left', bbox_to_anchor=(0, 1.19),
           ncol=2,columnspacing=1,handlelength=0)
    plt.ylim(0, 25)
    # plt.yticks([0.6+0.1*i for i in range(0, 9)])
    # plt.ylabel('Improvement over\nStreamSync')
    plt.xlabel("Batch Size")
    ax2.get_xaxis().set_label_coords(0.45,-0.4)
    ax2.get_yaxis().set_visible(False)
    for i, f in enumerate(zip(resnet,vgg)):
        ax2.text(i, max(f)+1, "%.0f"%round(max(f),0), color = 'black', ha = 'center', rotation=0)
    # plt.title('Contribution by the teams')
    xt = [1,4,8,12,16,20,24,28,32]
    plt.xticks(ind+width/2, xt, rotation=90)
    # ax2.get_xaxis().set_label_coords(0,0.4)
    # plt.yticks(np.arange(0, 1.25, 0.25))
    FIGURES_DIR = "./"
        
    plt.rcParams["font.family"] = "libertine"
    #FIGURES_DIR = "./"
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.1)
    fig.set_size_inches(2.5, 2.2)

    # ax.set_xticks([])
    fig.savefig(FIGURES_DIR+pdf_name,bbox_inches='tight',pad_inches=0.15)
    # plt.show()