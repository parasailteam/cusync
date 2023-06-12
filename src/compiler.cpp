#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <ostream>

class Dimension {
  std::string name_;
  uint lower_;
  uint upper_;
  
public:
  std::string name() const {return name_;}
  uint lower()       const {return lower_;}
  uint upper()       const {return upper_;}

  Dimension(std::string name, uint lower, uint upper) : 
    name_(name), lower_(lower), upper_(upper) {}

  std::pair<Dimension, Dimension> split(uint splitVal) {
    assert(splitVal > lower() && splitVal < upper());
    return std::make_pair(Dimension(name(), lower(), splitVal), 
                          Dimension(name(), splitVal, upper()));
  }

  void print(std::ostream& os) {
    os << "[" << name() << " = " << lower() << "->" << upper()  << "]";
  }

  struct Hash {
    size_t operator()(const Dimension &d) const {
      return std::hash<std::string>{}(d.name());
    }
  };
};

class Tile {
  typedef std::map<Dimension, size_t, Dimension::Hash> DimensionToSizeMap;
  DimensionToSizeMap dimSizes_;

  public:  
  uint size(std::string dim) {
    return dimSizes_[Dimension(dim, 0, 0)];
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
    dimSizes_.erase(Dimension(dim, 0, 0));
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
  virtual Grid* batchTiles(std::string dim, uint batch) = 0;
  virtual Grid* collapseDim(std::string dim) = 0;
  virtual Grid* split(std::string dim, uint splitValue) = 0;
  virtual void print(std::ostream& os) = 0;
};

class FullGrid : public Grid {
  typedef std::map<std::string, Dimension> NameToDimensionMap;
  NameToDimensionMap dims_;
  Tile tile_;

public:
  Tile tile() {return tile_;}

  FullGrid(std::vector<Dimension> dims, Tile tile) : 
    tile_(tile) {
    for (auto iter : dims) {
      dims_[iter.name()] = iter;
    }
  }

  FullGrid(NameToDimensionMap dims, Tile tile) : 
    tile_(tile), dims_(dims) {}
  
  Grid* batchTiles(std::string dimName, uint batch) {
    return new FullGrid(dims_, tile_.batch(dimName, batch));
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

  Grid* batchTiles(std::string dimName, uint batch) {
    std::vector<FullGrid*> batchedSubGrids;

    for (auto subGrid : subGrids_) {
      batchedSubGrids.push_back(dynamic_cast<FullGrid*>(subGrid->batchTiles(dimName, batch)));
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

