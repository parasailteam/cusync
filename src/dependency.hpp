#include <memory>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <ostream>
#include <algorithm>

#pragma once 

class ExprImpl {
public:
  ExprImpl(){}
  virtual ~ExprImpl() {}
};

template<typename T>
class ConstImpl : public ExprImpl {
  T val_;
public:
  T val() const {return val_;}
  ConstImpl(T val) : val_(val) {}
};

class VarImpl : public ExprImpl {
  std::string name_;
public:
  VarImpl(std::string name) : name_(name) {}
};

class BinaryExprImpl : public ExprImpl {
  BinaryExpr::BinaryOp op_;
  std::shared_ptr<ExprImpl> op1_;
  std::shared_ptr<ExprImpl> op2_;

public:
  BinaryExprImpl(std::shared_ptr<ExprImpl> op1, BinaryExpr::BinaryOp op, std::shared_ptr<ExprImpl> op2) : 
    op1_(op1), op_(op), op2_(op2) {}
};

class ForAllImpl : public ExprImpl {
  std::shared_ptr<DimensionImpl> var_;
  std::shared_ptr<ExprImpl> baseExpr_;
  int lower_;
  int upper_;

public:
  ForAllImpl(std::shared_ptr<DimensionImpl> var, std::shared_ptr<ExprImpl> baseExpr, uint lower, uint upper) : 
    var_(var), baseExpr_(baseExpr), lower_(lower), upper_(upper)
   {}
};

class ComputeTileImpl : public ExprImpl {
  std::vector<std::shared_ptr<ExprImpl>> dims_;

public:
  ComputeTileImpl(std::vector<std::shared_ptr<ExprImpl>> dims) : dims_(dims) {
  }
};

class DimensionImpl : public ExprImpl {
  std::string name_;
  uint lower_;
  uint upper_;
  
public:
  std::string name() const {return name_;}
  uint lower()       const {return lower_;}
  uint upper()       const {return upper_;}

  DimensionImpl(std::string name, uint lower, uint upper) : 
    name_(name), lower_(lower), upper_(upper) {
      assert(upper > lower);
    }

  std::pair<DimensionImpl, DimensionImpl> split(uint splitVal) {
    assert(splitVal > lower() && splitVal < upper());
    return std::make_pair(DimensionImpl(name(), lower(), splitVal), 
                          DimensionImpl(name(), splitVal, upper()));
  }
  
  uint size() {return upper() - lower();}
  
  void print(std::ostream& os) {
    os << "[" << name() << " = " << lower() << "->" << upper()  << "]";
  }

  struct Hash {
    size_t operator()(const DimensionImpl &d) const {
      return std::hash<std::string>{}(d.name());
    }
  };

  struct Comparer {
    bool operator()(const DimensionImpl& d, const DimensionImpl& e) const {
      return d.name() == e.name(); 
    }
  };
};