#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <ostream>
#include <algorithm>
#include <memory>
#include <sstream>

#include "dependency.hpp"
#include "dependencyast.hpp"
#include "dependencyvisitor.hpp"
#include "utils.h"

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
  virtual void codegen(std::ostream& os, int indent) = 0;
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

  void codegen(std::ostream& os, int indent) {
    os << indentStr(indent) << "if (";
    for (auto iter = dims_.begin(); iter != dims_.end();) {
      iter->second.genCondition(os);
      iter++;
      if (iter != dims_.end()) {
        os << " && ";
      }
    }
    os << ") {" << std::endl;
    //Consider src tiles {x,A1y+B1} and {x, A2y+B2}
    //If the tiles are batched then both tiles should share the same semaphore, i.e.,
    //if (y >= B1 && (y - B1) %A1 == 0) then return (y-B1)/A1; if (y >= B2 && (y-B2)%A2 == 0) then return (y-B2)/A2;
    //Otherwise if they are not batched then just return y;
    //TODO: Use __builtin_assume in Clang or __builtin_unreachable in gcc
    for (uint i = 0; i < dep_.srcTiles().size(); i++) {
      auto t = dep_.srcTiles()[i];
      if (t->isComputeTile()) {
        std::dynamic_pointer_cast<ComputeTileImpl>(t)->genTileIndex(os, indent+1, batch_ > 1 && batch_ > i);
        os << std::endl;
      } else if (t->isForAll()) {
        auto fa = std::dynamic_pointer_cast<ForAllImpl>(t);
        auto c = std::dynamic_pointer_cast<ComputeTileImpl>(fa->baseExpr());
        //Create two new tiles:
        //{x, y} => {x, batch * y + 0}, {x, batch*y + 1}, {x + batch*y + 2}, ...
        //then generate code for each of them
        if (batch_ > 1) {
          std::vector<std::shared_ptr<ComputeTileImpl>> newTiles;
          for (uint b = 0; b < batch_; b++) {
            std::shared_ptr<ComputeTileImpl> newTile = c->newTile(1, batch_, b); //TODO: hardcoding dim 1 
            newTiles.push_back(newTile);
          }
          for (auto newTile : newTiles) {
            newTile->genTileIndex(os, indent+1, true);
            os << std::endl;
          }
        } else {
          c->genTileIndex(os, indent+1, batch_ > 1 && batch_ > i);
          os << std::endl;
        }
      } else {
        std::cout << "Invalid " << typeid(*t.get()).name() << std::endl;
        assert(false);
      }
      if (batch_ == 1) break;
    }
    os << indentStr(indent) << "}";
  }

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
      allSubGridCasesNext.clear();
      if (gIter == allBatchedGrids.begin()) {
        for (auto g : gIter->second) {
          std::vector<FullGrid*> v = {(FullGrid*)g};
          allSubGridCasesNext.push_back(v);
        }
      } else {
        for (auto gridVec : allSubGridCasesPrev) {
          for (auto g : gIter->second) {
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

  void codegen(std::ostream& os, int indent) {
    for (auto sg : subGrids_) {
      sg->codegen(os, indent+1);
      os << std::endl;
    }
  }

  void print(std::ostream& os) {
    os << "split{";
    for (auto subGrid : subGrids_) {
      subGrid->print(os);
      os << "; ";
    }
    os << "}";
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
    tileBatches = {2};
  } else {
    for (uint i = 2; i <= fullGrid->dep().srcTiles().size(); i++) {
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

  for (auto lastTB : lastTBInWave) {
    std::cout << "404: " << std::get<0>(lastTB) << " " << std::get<1>(lastTB) << std::endl;
    Grid* sg = fullGrid->split("x", std::get<0>(lastTB));
    std::cout << "after x ";
    sg->print(std::cout);
    std::cout << std::endl;
    sg = sg->split("k", std::get<1>(lastTB));
    std::cout << " after k ";
    sg->print(std::cout);
    std::cout << std::endl;
    allGridCases.push_back(sg);
    sg->batchGrid(allGridCases, tileBatches);
  }

  std::cout<<"number of grids: "<< allGridCases.size() << std::endl;
  //split values are these lastTBInWave index, so split the grid among these dimensions
  
  //Combine grids that are contiguous with same tile batch in all dimensions
  int c = 0;
  for (auto g : allGridCases) {
    if (std::string(typeid(*g).name()).find("SplitGrid") == -1)
      continue;
    c++;
    if (c > 10) break;
    std::stringstream ss;
    g->codegen(ss, 0);
    std::cout << "=====" << std::endl;
    g->print(std::cout);
    std::cout << std::endl;
    std::cout << ss.str() << std::endl;
  }
  }

int main(int argc, char* argv[]) {
  Dimension x("x", 0, 8);
  Dimension y("y", 0, 96);
  Dimension k("k", 0, 96);

  //TODO: 
  
  if (true) {
    ComputeTile dstTile({x, y});
    ComputeTile srcTile({x, k});
    ForAll allSrcTiles (k, srcTile, 0, 96);
    Dependency dep = Dependency(allSrcTiles, dstTile);
    FullGrid fg(std::vector<DimensionImpl>({*x.impl(), *k.impl()}), dep);
    search(&fg);
  }

  if (false) {
    //Determine the upper bound and lower bound of srcTiles by adding y.upper.
    //Create dependency vectors between each pair of src tiles.
    //Apply those vectors one by one and decrease the grid.
    //Consider src tiles {x,A1y+B1} and {x, A2y+B2} can be synchronized only if the set produce distinct values for each y, i.e., A1 and A2 are coprime
    
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

  if (false) {
    Dimension y("y", 0, 32);
    ComputeTile dstTile({x, y});
    ComputeTile srcTile1({x, 3*y});
    ComputeTile srcTile2({x, 3*y + 1});
    ComputeTile srcTile3({x, 3*y + 2});

    //Assuming that the src tiles are added in the order of how they are accessed
    Dependency dep({srcTile1, srcTile2, srcTile3}, dstTile);
    FullGrid fg(std::vector<DimensionImpl>({*x.impl(), *k.impl()}), dep);
    search(&fg);

    return 0;
  }

  // FullGrid fg(std::vector<DimensionImpl>({*x, *k}), Tile({*x, *k}, {1,1}));
  // search(&fg);
  return 0;
}