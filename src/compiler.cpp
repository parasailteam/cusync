#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <ostream>
#include <algorithm>

template<typename T> 
T DIVUP(T x, T y) {
  return ((x + y - 1)/y);
}

template<typename T> 
T ROUNDUP(T x, T y) {
  return DIVUP(x, y) * y;
}

class Dimension {
  std::string name_;
  uint lower_;
  uint upper_;
  
public:
  std::string name() const {return name_;}
  uint lower()       const {return lower_;}
  uint upper()       const {return upper_;}

  Dimension(std::string name, uint lower, uint upper) : 
    name_(name), lower_(lower), upper_(upper) {
      assert(upper > lower);
    }

  std::pair<Dimension, Dimension> split(uint splitVal) {
    assert(splitVal > lower() && splitVal < upper());
    return std::make_pair(Dimension(name(), lower(), splitVal), 
                          Dimension(name(), splitVal, upper()));
  }
  
  uint size() {return upper() - lower();}
  
  void print(std::ostream& os) {
    os << "[" << name() << " = " << lower() << "->" << upper()  << "]";
  }

  struct Hash {
    size_t operator()(const Dimension &d) const {
      return std::hash<std::string>{}(d.name());
    }
  };

  struct Comparer {
    bool operator()(const Dimension& d, const Dimension& e) const {
      return d.name() == e.name(); 
    }
  };
};

class Tile {
  typedef std::map<Dimension, size_t, Dimension::Comparer> DimensionToSizeMap;
  DimensionToSizeMap dimSizes_;

  public:  
  uint size(std::string dim) {
    Dimension d = Dimension(dim, 0, 0);
    return dimSizes_.at(d);
  }

  Tile(std::vector<Dimension> dims, std::vector<uint> sizes) {
    assert(dims.size() == sizes.size());
    for (uint i = 0; i < dims.size(); i++) {
      dimSizes_[dims[i]] = sizes[i];
    }
  }

  Tile(DimensionToSizeMap dimSizes) : dimSizes_(dimSizes) {}

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
    Dimension d = Dimension(dim, 0, 0);
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
public:
  typedef std::map<std::string, Dimension> NameToDimensionMap;

protected:
  Grid(){}
public:
  virtual NameToDimensionMap dims() = 0;
  virtual void batchGrid(std::vector<Grid*>& output, std::vector<uint> batchSizes) = 0;
  virtual Grid* batchDim(std::string dim, uint batch) = 0;
  virtual Grid* collapseDim(std::string dim) = 0;
  virtual Grid* split(std::string dim, uint splitValue) = 0;
  virtual void print(std::ostream& os) = 0;
};

class FullGrid : public Grid {
private:
  NameToDimensionMap dims_;
  Tile tile_;

public:
  Tile tile()               {return tile_;}
  NameToDimensionMap dims() {return dims_;}
  
  FullGrid(std::vector<Dimension> dims, Tile tile) : 
    tile_(tile) {
    for (auto iter : dims) {
      dims_.emplace(iter.name(), iter);
    }
  }

  FullGrid(NameToDimensionMap dims, Tile tile) : 
    tile_(tile), dims_(dims) {}
  
  Grid* batchDim(std::string dimName, uint batch) {
    return new FullGrid(dims_, tile_.batch(dimName, batch));
  }

  void batchGrid(std::vector<Grid*>& output, std::vector<uint> batchSizes) {
    auto dimIter = dims_.begin();
    std::vector<Grid*> newGrids;
    while (dimIter != dims_.end()) {
      if (newGrids.empty()) {
        for (auto batch : batchSizes) {
          newGrids.push_back(batchDim(dimIter->first, batch));
        }
      } else {
        size_t numGrids = newGrids.size();
        for (int i = 0; i < numGrids; i++) {
          for (auto batch : batchSizes) {
            newGrids.push_back(newGrids[i]->batchDim(dimIter->first, batch));
          }
        }
      }
      dimIter++;
    }

    for (auto newGrid : newGrids) {
      output.push_back(newGrid);
    }
  }

  Grid* collapseDim(std::string dimName) {
    NameToDimensionMap newDim = dims_;
    newDim.erase(dimName);
    Tile tile = tile_.eraseDim(dimName);

    return new FullGrid(newDim, tile);
  }

  Grid* split(std::string dimName, uint splitValue);

  void print(std::ostream& os) {
    os << "{";
    for (auto iter : dims_) {
      iter.second.print(os);
      os << ",";
    }
    os << " tile = ";
    tile().print(os);
    os << "}";
  }
};

class SplitGrid : public Grid {
  std::vector<FullGrid*> subGrids_;

public:
  SplitGrid(std::vector<FullGrid*> subGrids) : subGrids_(subGrids) {}
  NameToDimensionMap dims() {return subGrids_[0]->dims();}
  
  Grid* batchDim(std::string dimName, uint batch) {
    std::vector<FullGrid*> batchedSubGrids;

    for (auto subGrid : subGrids_) {
      batchedSubGrids.push_back(dynamic_cast<FullGrid*>(subGrid->batchDim(dimName, batch)));
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
  std::vector<Dimension> dims1, dims2;

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

  return new SplitGrid(std::vector<FullGrid*>({new FullGrid(dims1, tile()), new FullGrid(dims2, tile())}));
}

void search(FullGrid* fullGrid) {
  //Split a grid only twice
  //Assuming dependent matrix multiplications for now
  std::vector<uint> tileBatches = {2, 4, 8, 16};
  std::vector<uint> splitValues = {};
  uint tbsPerSM = 2;
  uint NumSMs = 80;
  uint tbsPerWave = tbsPerSM * NumSMs;

  //Find possible split values
  auto dims = fullGrid->dims();
  uint totalTBs = dims.at("x").size() * dims.at("y").size();
  //assume row major order and totalTbs > tbsPerWave
  uint waves = DIVUP(totalTBs, tbsPerWave);

  std::vector<std::tuple<uint, uint, uint>> lastTBInWave;
  for (uint y = 0; y < dims.at("y").size(); y++) {
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
  Dimension x = Dimension("x", 0, 8);
  Dimension y = Dimension("y", 0, 96);
  
  FullGrid fg(std::vector<Dimension>({x, y}), Tile({x, y}, {1,1}));
  search(&fg);
  return 0;
}