#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <ostream>
#include <algorithm>
#include <memory>

#include "dependency.hpp"
#include "dependencyast.hpp"
#include "dependencyvisitor.hpp"

template<typename T> 
T DIVUP(T x, T y) {
  return ((x + y - 1)/y);
}

template<typename T> 
T ROUNDUP(T x, T y) {
  return DIVUP(x, y) * y;
}

//This tile is a batch of Grid not computation tile
class Tile {
  typedef std::map<DimensionImpl, size_t, DimensionImpl::Comparer> DimensionToSizeMap;
  DimensionToSizeMap dimSizes_;

  public:  
  uint size(std::string dim) {
    DimensionImpl d = DimensionImpl(dim, 0, 0);
    return dimSizes_.at(d);
  }

  Tile(std::vector<DimensionImpl> dims, std::vector<uint> sizes) {
    assert(dims.size() == sizes.size());
    for (uint i = 0; i < dims.size(); i++) {
      dimSizes_[dims[i]] = sizes[i];
    }
  }

  Tile(DimensionToSizeMap dimSizes) : dimSizes_(dimSizes) {}

  Tile batch(uint batchSize) {
  }

  Tile batch(std::string dim, uint batchSize) {
    DimensionToSizeMap newDimSizes;
    for (auto iter : dimSizes_) {
      if (iter.first.name() == dim) {
        newDimSizes[iter.first] = batchSize;
      } else {
        newDimSizes[iter.first] = iter.second;
      }
    }
    
    return Tile(newDimSizes);
  }

  Tile eraseDim(std::string dim) {
    DimensionImpl d = DimensionImpl(dim, 0, 0);
    dimSizes_.erase(d);
  }

  void print(std::ostream& os) {
    os << "[";
    for (auto iter : dimSizes_) {
      os << iter.first.name() << "/" << iter.second << ", ";
    }
    os << "]";
  }
};


class FullGrid;
class SplitGrid;

class Grid {
protected:
  Grid(){}

public:
  virtual NameToDimensionMap dims() = 0;
  virtual void batchGrid(std::vector<Grid*>& output, std::vector<uint> batchSizes) = 0;
  virtual Grid* batchDim(uint batch) = 0;
  virtual Grid* collapseDim(std::string dim) = 0;
  virtual Grid* split(std::string dim, uint splitValue) = 0;
  virtual void print(std::ostream& os) = 0;
};

class FullGrid : public Grid {
private:
  NameToDimensionMap dims_;
  Dependency dep_;
  uint batch_;

  FullGrid(NameToDimensionMap dims, Dependency dep, uint batch) : 
    dims_(dims), dep_(dep), batch_(batch) {}
  FullGrid(std::vector<DimensionImpl> dims, Dependency dep, uint batch) : 
  dep_(dep), batch_(batch) {
    for (auto iter : dims) {
      dims_.emplace(iter.name(), iter);
    }
  }
  
public:
  NameToDimensionMap dims() {return dims_;}
  Dependency dep()          {return dep_;}
  FullGrid(std::vector<DimensionImpl> dims, Dependency dep) : 
    dep_(dep), batch_(1) {
    //Check that range of dims is same as range of dims in dep
    // TODO:
    // checkDimAndDepSizes(dims, dep);
    for (auto iter : dims) {
      dims_.emplace(iter.name(), iter);
    }
  }

  void checkDimAndDepSizes(std::vector<DimensionImpl> dims, Dependency dep) {
    for (auto srcTile : dep.srcTiles()) {
      for (uint dim = 0; dim < dims.size(); dim++) {
        auto dstTile = dep.dstTile();
        auto srcDim = std::dynamic_pointer_cast<DimensionImpl>(dstTile->dims()[dim])->name();
        ComputeBoundsOfTile visitor(srcDim);
        srcTile->visit(visitor);
        uint maxValue = visitor.maxValue();
        uint minValue = visitor.minValue();
        assert(maxValue == dims[dim].upper());
        assert(minValue == dims[dim].lower());
      }
    }
  }

  std::pair<uint, uint> getTileAccessCoeff(ComputeTileImpl& tile, uint idx) {
    auto dimExpr = tile.dims()[idx];
    ComputeExprValue visitor0(0);
    dimExpr->visit(visitor0);
    uint adder = visitor0.computedValue();

    ComputeExprValue visitor1(1);
    dimExpr->visit(visitor1);
    uint v = visitor1.computedValue();
    uint coeff = v - adder;

    return std::make_pair(coeff, adder);
  }

  Grid* batchDim(uint batch) {
    return new FullGrid(dims_, dep_, batch_*batch);
  }

  void batchGrid(std::vector<Grid*>& output, std::vector<uint> batchSizes) {
    // Batch tiles one by one
    for (auto it : batchSizes) {
      output.push_back(batchDim(it));
    }
    
    // Following code generates all cases by batching tiles along all dimensions
    // auto dimIter = dims_.begin();
    // std::vector<Grid*> newGrids;
    // while (dimIter != dims_.end()) {
    //   if (newGrids.empty()) {
    //     for (auto batch : batchSizes) {
    //       newGrids.push_back(batchDim(dimIter->first, batch));
    //     }
    //   } else {
    //     size_t numGrids = newGrids.size();
    //     for (int i = 0; i < numGrids; i++) {
    //       for (auto batch : batchSizes) {
    //         newGrids.push_back(newGrids[i]->batchDim(dimIter->first, batch));
    //       }
    //     }
    //   }
    //   dimIter++;
    // }

    // for (auto newGrid : newGrids) {
    //   output.push_back(newGrid);
    // }
  }

  Grid* collapseDim(std::string dimName) {
    NameToDimensionMap newDim = dims_;
    newDim.erase(dimName);
    // Tile tile = tile_.eraseDim(dimName);

    return new FullGrid(newDim, dep_, batch_);
  }

  Grid* split(std::string dimName, uint splitValue);

  void print(std::ostream& os) {
    os << "{";
    for (auto iter : dims_) {
      iter.second.print(os);
      os << ",";
    }
    os << " batch = ";
    os << batch_;
    os << "}";
  }
};

class SplitGrid : public Grid {
  std::vector<FullGrid*> subGrids_;

public:
  SplitGrid(std::vector<FullGrid*> subGrids) : subGrids_(subGrids) {}
  NameToDimensionMap dims() {return subGrids_[0]->dims();}
  
  Grid* batchDim(uint batch) {
    std::vector<FullGrid*> batchedSubGrids;

    for (auto subGrid : subGrids_) {
      batchedSubGrids.push_back(dynamic_cast<FullGrid*>(subGrid->batchDim(batch)));
    }

    return new SplitGrid(batchedSubGrids);
  }

  Grid* collapseDim(std::string dimName) {
    std::vector<FullGrid*> batchedSubGrids;

    for (auto subGrid : subGrids_) {
      batchedSubGrids.push_back(dynamic_cast<FullGrid*>(subGrid->collapseDim(dimName)));
    }

    return new SplitGrid(batchedSubGrids);
  }

  Grid* split(std::string dimName, uint splitValue) {
    std::vector<FullGrid*> batchedSubGrids;

    for (auto subGrid : subGrids_) {
      SplitGrid* newSplit = dynamic_cast<SplitGrid*>(subGrid->split(dimName, splitValue));
      batchedSubGrids.insert(batchedSubGrids.end(), newSplit->subGrids_.begin(), newSplit->subGrids_.end());
      delete newSplit;
    }

    return new SplitGrid(batchedSubGrids);
  }

  void batchGrid(std::vector<Grid*>& output, std::vector<uint> batchSizes) {
    std::map<FullGrid*, std::vector<Grid*>> allBatchedGrids;

    for (auto g : subGrids_) {
      std::vector<Grid*> batchedGrids;
      g->batchGrid(batchedGrids, batchSizes);
      batchedGrids.push_back(g);
      allBatchedGrids.emplace(g, batchedGrids);
    }

    //Create SplitGrid for all cases
    std::vector<Grid*> newSplitGrids;
    std::vector<std::vector<FullGrid*>> allSubGridCasesPrev;
    std::vector<std::vector<FullGrid*>> allSubGridCasesNext;

    auto gIter = allBatchedGrids.begin();
    while (gIter != allBatchedGrids.end()) {
      if (gIter == allBatchedGrids.begin()) {
        std::vector<FullGrid*> v = std::vector<FullGrid*>();
        for (auto g : gIter->second) {
          v.push_back((FullGrid*)g);
        }
        allSubGridCasesNext.push_back(v);
      } else {
        for (auto g : gIter->second) {
          for (auto gridVec : allSubGridCasesPrev) {
            std::vector<FullGrid*> newVec = gridVec;
            // std::transform(gridVec.begin(), gridVec.end(), newVec.begin(), 
            //                [](FullGrid* fg){return (FullGrid*)fg;});
            newVec.push_back((FullGrid*)g);
            allSubGridCasesNext.push_back(newVec);
          }
        }
      }
      allSubGridCasesPrev = allSubGridCasesNext;
      gIter++;
    }

    for (auto gridVec : allSubGridCasesNext) {
      output.push_back(new SplitGrid(gridVec));
    }
  }

  void print(std::ostream& os) {
    os << "split<";
    for (auto subGrid : subGrids_) {
      subGrid->print(os);
      os << "; ";
    }
    os << ">";
  }
};

Grid* FullGrid::split(std::string dimName, uint splitValue) {
  std::vector<DimensionImpl> dims1, dims2;

  for (auto iter : dims_) {
    if (iter.first == dimName) {
      auto splitDim = iter.second.split(splitValue);
      dims1.push_back(splitDim.first);
      dims2.push_back(splitDim.second);
    } else {
      dims1.push_back(iter.second);
      dims2.push_back(iter.second);
    }
  }

  return new SplitGrid(std::vector<FullGrid*>({new FullGrid(dims1, dep_, batch_),
                                               new FullGrid(dims2, dep_, batch_)}));
}

void search(FullGrid* fullGrid) {
  //Split a grid only twice
  //Assuming dependent matrix multiplications for now
  std::vector<uint> tileBatches;
  if (fullGrid->dep().srcTiles().size() == 1 && fullGrid->dep().srcTiles()[0]->isForAll()) {
    tileBatches = {2, 4, 8, 16};
  } else {
    for (uint i = 1; i < fullGrid->dep().srcTiles().size(); i++) {
      tileBatches.push_back(i); 
    }
  }

  std::vector<uint> splitValues = {};
  uint tbsPerSM = 2;
  uint NumSMs = 80;
  uint tbsPerWave = tbsPerSM * NumSMs;

  //Find possible split values
  auto dims = fullGrid->dims();
  uint totalTBs = 1;
  for (auto dim : dims) {
    totalTBs *= dim.second.size();
  }
  
  //assume row major order and totalTbs > tbsPerWave
  uint waves = DIVUP(totalTBs, tbsPerWave);

  std::vector<std::tuple<uint, uint, uint>> lastTBInWave;
  
  //TODO: Fix these dim of "k" and "x"
  for (uint y = 0; y < dims.at("k").size(); y++) {
    for (uint x = 0; x < dims.at("x").size(); x++) {
      uint tb = y * dims.at("x").size() + x;
      if ((tb+1)%tbsPerWave == 0) {
        lastTBInWave.push_back(std::make_tuple(x, y, tb));
      }
    }
  }

  std::vector<Grid*> allGridCases;
  allGridCases.push_back(fullGrid);
  
  fullGrid->batchGrid(allGridCases, tileBatches);

  std::vector<SplitGrid*> allSplitGrids;
  for (auto lastTB : lastTBInWave) {
    Grid* sg = fullGrid->split("x", std::get<0>(lastTB));
    sg = sg->split("y", std::get<1>(lastTB));
    allGridCases.push_back(sg);
    sg->batchGrid(allGridCases, tileBatches);
  }

  std::cout<<"number of grids: "<< allGridCases.size() << std::endl;
  //split values are these lastTBInWave index, so split the grid among these dimensions
  
  //Combine grids that are contiguous with same tile batch in all dimensions

  //
}

int main(int argc, char* argv[]) {
  Dimension x("x", 0, 8);
  Dimension y("y", 0, 96);
  Dimension k("k", 0, 96);

  {
    ComputeTile dstTile({x, y});
    ComputeTile srcTile({x, k});
    ForAll allSrcTiles (k, srcTile, 0, 96);
    Dependency dep = Dependency(allSrcTiles, dstTile);
    FullGrid fg(std::vector<DimensionImpl>({*x.impl(), *k.impl()}), dep);
    search(&fg);
  }

  {
    //Determine the upper bound and lower bound of srcTiles by adding y.upper.
    //Create dependency vectors between each pair of src tiles.
    //Apply those vectors one by one and decrease the grid.
    //src tiles {x,A1y+B1} and {x, A2y+B2} can be synchronized only if the set produce distinct values for each y, i.e., A1 and A2 are coprime
    
    //To combine both tiles with B1=B2=0, if y%A1 == 0 then y/A1 and if y%A2 == 0 then y/A2 
    //When A1=A2=1, if (y>=B1) then y-B1 else if y >= B2 then y - B2
    //For arbitrary A1, A2, B1, B2, if (y >= B1 && (y - B1) %A1 == 0) then return (y-B1)/A1; if (y >= B2 && (y-B2)%A2 == 0) then return (y-B2)/A2
    
    //Batched A1y+B1 tiles with each other
    //return ((y-B1)/A1)/T1
    //Or Batch A1y+B1 and A2y+B2 tiles with each other

    //Split grid works just like it does normally

    Dimension y("y", 0, 32);
    ComputeTile dstTile({x, y});
    ComputeTile srcTile1({x, y});
    ComputeTile srcTile2({x, y + 32});
    ComputeTile srcTile3({x, y + 2*32});

    Dependency dep({srcTile1, srcTile2, srcTile3}, dstTile);
    FullGrid fg(std::vector<DimensionImpl>({*x.impl(), *k.impl()}), dep);
    search(&fg);
  }

  {
    Dimension y("y", 0, 32);
    ComputeTile dstTile({x, y});
    ComputeTile srcTile1({x, 3*y});
    ComputeTile srcTile2({x, 3*y + 1});
    ComputeTile srcTile3({x, 3*y + 2});

    //Assuming that the src tiles are added in the order of how they are accessed
    Dependency dep({srcTile1, srcTile2, srcTile3}, dstTile);
    FullGrid fg(std::vector<DimensionImpl>({*x.impl(), *k.impl()}), dep);
    search(&fg);
  }

  // FullGrid fg(std::vector<DimensionImpl>({*x, *k}), Tile({*x, *k}, {1,1}));
  // search(&fg);
  return 0;
}