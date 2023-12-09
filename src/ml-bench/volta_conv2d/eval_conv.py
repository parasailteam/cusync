import torch
import subprocess
from statistics import stdev
import re 
import json
import os
import sys

if len(sys.argv) < 2:
  print ("Arguments are: resnet|vgg")
  sys.exit(0)

resnet_or_vgg = sys.argv[1].strip()
if resnet_or_vgg not in ['resnet', 'vgg']:
  print ("Arguments are: resnet|vgg")
  sys.exit(0)

#Image Height/Width for a Conv kernel size
hw = {
    64: {"h": 56, "w": 56},
    128: {"h": 28, "w": 28},
    256: {"h": 14, "w": 14},
    512: {"h": 7, "w": 7}
}

#Tile sizes for policies, batch sizes, and kernel size
tiles = {
    1:  {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":2},
              "rowsync": {"split_k":2, 
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": False,
                          "NoAtomicAdd": True},
             },
        128: {"TileSizes": [64, 64, 32, 32, 32, 32],
              "baseline": {"split_k":2},
              "rowsync":  {"split_k":2,
                           "AvoidCustomOrder": True,
                           "AvoidWaitKernel": True},
              "tilesync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True},
            },
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":4},
              "rowsync": {"split_k":4, 
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True},
             },
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":8},
              "rowsync": {"split_k":8},
              "tilesync": {"split_k":8,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True},
             }
        },
    4:  {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":2},
              "rowsync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,},
              "tilesync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": False,
                          "NoAtomicAdd": True}
             },
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":3},
              "rowsync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":2},
              "rowsync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        512: {"TileSizes": [64, 64, 32, 32, 32, 32],
              "baseline": {"split_k":4},
              "rowsync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             }
        },
    8:  {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": False,
                          "NoAtomicAdd": True}
             },
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        256: {"TileSizes": [64, 256, 32, 32, 128, 32], 
              "baseline": {"split_k":4},
              "rowsync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              },
        512: {"TileSizes": [64, 128, 32, 32, 64, 32], 
              "baseline": {"split_k":4},
              "rowsync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              },
        },
    12: {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": False,
                          "NoAtomicAdd": True}
              },
        128: {"TileSizes": [128, 128, 32, 64, 64, 32],
              "baseline": {"split_k":1},
              "rowsync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":2,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
            },
        256: {"TileSizes": [64, 256, 32, 32, 128, 32], 
              "baseline": {"split_k":3},
              "rowsync": {"split_k":3},
              "tilesync": {"split_k":3,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              },
        512: {"TileSizes": [64, 256, 32, 32, 128, 32], 
              "baseline": {"split_k":4},
              "rowsync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              },
    },
    16: {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        128: {"TileSizes": [64, 256, 32, 32, 128, 32],
              "baseline": {"split_k":3},
              "rowsync": {"split_k":3,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,},
              "tilesync": {"split_k":3,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":3},
              "rowsync": {"split_k":3},
              "tilesync": {"split_k":3,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              },
        512: {"TileSizes": [64, 128, 32, 32, 64, 32],
              "baseline": {"split_k":3},
              "rowsync": {"split_k":3},
              "tilesync": {"split_k":3,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              }
        },
    20: {64: {"TileSizes": [256, 64, 32, 128, 32, 32],
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              },
        128: {"TileSizes": [128, 128, 32, 64, 64, 32],
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
              },
        256: {"TileSizes": [64, 256, 32, 32, 128, 32],
              "baseline": {"split_k":2},
              "rowsync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        512: {"TileSizes": [128, 128, 32, 64, 64, 32],
              "baseline": {"split_k":4},
              "rowsync": {"split_k":4, 
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True},
              "tilesync": {"split_k":4,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             }
        },
    24: {64: {"TileSizes": [256, 64, 32, 128, 32, 32],
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        128: {"TileSizes": [128, 128, 32, 64, 64, 32], 
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        256: {"TileSizes": [64, 256, 32, 32, 128, 32], 
              "baseline": {"split_k":2},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": True,
                          "AvoidWaitKernel": True,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        512: {"TileSizes": [64, 64, 32, 32, 32, 32],
              "baseline": {"split_k":3},
              "rowsync": {"split_k":3},
              "tilesync": {"split_k":3,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        },
    28: {64: {"TileSizes": [256, 64, 32, 128, 32, 32],
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        128: {"TileSizes": [128, 128, 32, 64, 64, 32], 
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        256: {"TileSizes": [64, 256, 32, 32, 128, 32], 
              "baseline": {"split_k":3},
              "rowsync": {"split_k":2},
              "tilesync": {"split_k":2,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":3},
              "rowsync": {"split_k":3},
              "tilesync": {"split_k":3,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        },
    32: {64: {"TileSizes": [256, 64, 32, 128, 32, 32],
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        128: {"TileSizes": [128, 128, 32, 64, 64, 32], 
              "baseline": {"split_k":1},
              "rowsync": {"split_k":1},
              "tilesync": {"split_k":1,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        256: {"TileSizes": [64, 256, 32, 32, 128, 32],
              "baseline": {"split_k":3},
              "rowsync": {"split_k":2},
              "tilesync": {"split_k":2,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             },
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k":3},
              "rowsync": {"split_k":3},
              "tilesync": {"split_k":3,
                          "AvoidCustomOrder": False,
                          "AvoidWaitKernel": False,
                          "ReorderTileLoads": True,
                          "NoAtomicAdd": True}
             }
    }
}

def exec_command(command):
  '''Execute a command'''
  print(f"Executing {command} in {os.getcwd()}")
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print("Error ", o)

  return o

def getAllTimes(s, START, END):
  '''Parse output to obtain times'''
  alltimes = {}
  assert START in s
  assert END in s
  s = s[s.find(START):s.find(END)]
  s = s[s.find("\n"):]
  alljsons = []
  for l in re.findall(r".+", s):
    j = json.loads(l)
    alljsons += [j]
  
  def sortkey(elem):
    return elem["Total"]
  
  alljsons.sort(key=sortkey)
  p = 0.9
  alljsons = alljsons[:int(len(alljsons)*0.9)]
  for j in alljsons:
    for k in j:
      if k not in alltimes:
        alltimes[k] = [] 
      alltimes[k] += [float(j[k])]

  return alltimes

def avg(l):
  return sum(l)/len(l)


def slurp(path):
  with open(path, "r") as f:
    return f.read()

#Create build and results directory
def buildDir(f):
  return 'build/'+f

if not os.path.exists(buildDir('')):
  os.mkdir(buildDir(''))

def deleteFiles(syncPolicies):
  command = "rm -f "
  for policy in syncPolicies:
    command += buildDir("conv-eval-%s "%(policy))

  o = exec_command(command)

def makeFiles(syncPolicies):
  command = "make "
  for policy in syncPolicies:
    command += buildDir("conv-eval-%s "%(policy))

  flags = "-j"
  command += flags
  o = exec_command(command)


def genFiles(batchInfo, syncPolicy):
  '''Make temp source files for each sync policy for given tile sizes'''
  inFile = resnet_or_vgg + '.cu'
  outFile = buildDir("conv-eval-" + syncPolicy + ".cu")
  fileContents = slurp(inFile)
  tilesCode = """using ThreadblockShape = cutlass::gemm::GemmShape<%d, %d, %d>;
using WarpShape = cutlass::gemm::GemmShape<%d, %d, %d>;"""
  tilesCode = tilesCode % tuple(batchInfo["TileSizes"])
  tilesCodeStart = fileContents.find("//<eval tiles>") + len("//<eval tiles>")
  tilesCodeEnd = fileContents.find("//</eval tiles>")
  fileContents = fileContents[0:tilesCodeStart] + "\n" + tilesCode + "\n" + fileContents[tilesCodeEnd:]
  optimizationsStart = fileContents.find("//<OPTIMIZATIONS>") + len("//<OPTIMIZATIONS>")
  optimizationsEnd = fileContents.find("//</OPTIMIZATIONS>")
  optimizationsCode = ""
  batchInfo = batchInfo["tilesync"] if syncPolicy == 'baseline' else batchInfo[syncPolicy]
  if syncPolicy != 'baseline':
    if "AvoidCustomOrder" in batchInfo and batchInfo["AvoidCustomOrder"] == True:
      optimizationsCode += "#define AVOID_CUSTOM_ORDER"+"\n"
    else:
      optimizationsCode += "#undef AVOID_CUSTOM_ORDER"+"\n"
    if "AvoidWaitKernel" in batchInfo and batchInfo["AvoidWaitKernel"] == True:
      optimizationsCode += "#define AVOID_WAIT_KERNEL"+"\n"
    else:
      optimizationsCode += "#undef AVOID_WAIT_KERNEL"+"\n"
    if "ReorderTileLoads" in batchInfo and batchInfo["ReorderTileLoads"] == True:
      optimizationsCode += "#define REORDER_TILE_LOADS"+"\n"
    else:
      optimizationsCode += "#undef REORDER_TILE_LOADS"+"\n"
    if "NoAtomicAdd" in batchInfo and batchInfo["NoAtomicAdd"] == True:
      optimizationsCode += "#define NO_ATOMIC_ADD"+"\n"
    else:
      optimizationsCode += "#undef NO_ATOMIC_ADD"+"\n"

  optimizationsCode += "#define " + syncPolicy.upper() + "\n"
  optimizationsCode += "#define " + "EVAL_TILE_SIZES" + "\n"
  fileContents = fileContents[0:optimizationsStart] + "\n" + optimizationsCode + "\n" + fileContents[optimizationsEnd:]

  if os.path.exists(outFile):
    with open(outFile, "r") as f:
      oldContents = f.read()
      if fileContents == oldContents:
        return
  with open(outFile, "w") as f:
    f.write(fileContents)

results_csv = ""

#Delete temp files for all policies
policies=['rowsync'] #,'tilesync'
deleteFiles(policies+['baseline'])

#Run evaluation
for c in ([64,128,256,512] if resnet_or_vgg == 'resnet' else [256,512]): #
  for m in [1, 4, 8,12, 16, 20, 24, 28, 32]:
    command_args = f"--n={m} --h={hw[c]['h']} --w={hw[c]['w']} --c={c} --k={c} --r=3 --s=3"
    split_k = f"--split_k_slices={tiles[m][c]['baseline']['split_k']}"
    policies = ['rowsync', 'tilesync']
    for syncPolicy in (policies+['baseline']):
      genFiles(tiles[m][c], syncPolicy)

    #Create and make all policies
    makeFiles(policies+['baseline'])

    #Evaluate baseline
    o = exec_command(buildDir("./conv-eval-baseline ") + command_args + " " + split_k)
    baselineTimes = getAllTimes(o, "START-BASELINE", "END-BASELINE")
    bTimes = baselineTimes["Total"]
    results_csv += f"{m} & {c} & baseline & {'%.2f'%avg(bTimes)} & {'%.2f'%stdev(bTimes)}\n"

    #Evaluate each policy
    for syncType in policies:
      split_k = f"--split_k_slices={tiles[m][c][syncType]['split_k']}"
      o = exec_command(buildDir(f"./conv-eval-{syncType} ") + command_args + " " + split_k)
      overlapTimes = getAllTimes(o, "START-OVERLAP", "END-OVERLAP")
      oTimes = overlapTimes["Total"]
      results_csv += f"{m} & {c} & {syncType} & {'%.2f'%avg(bTimes)} & {'%.2f'%stdev(bTimes)} & {'%.2f'%avg(oTimes)} & {'%.2f'%(stdev(oTimes))} & {'%.2f'%((avg(bTimes)-avg(oTimes))*100/avg(bTimes))}\n"

if not os.path.exists("results"):
  os.mkdir("results")

with open(f"results/{resnet_or_vgg}.csv", "w") as f:
  f.write(results_csv)