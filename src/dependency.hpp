#include <memory>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <ostream>
#include <algorithm>
#include <stack>

#pragma once 

class ExprImpl;
class VarImpl;
class BinaryExprImpl;
class ForAllImpl;
class DimensionImpl;
class UIntConstImpl;
class ComputeTileImpl;
class Visitor;

enum BinaryOp {
  Add,
  Sub,
  Mul,
  Div
};

typedef std::map<std::string, DimensionImpl> NameToDimensionMap;


class Visitor {
public:
  Visitor() {}
  virtual void visit(ExprImpl& expr);
  virtual void visit(BinaryExprImpl& expr) = 0;
  virtual void visit(DimensionImpl& expr) = 0;
  virtual void visit(UIntConstImpl& c) = 0;
  virtual void visit(ForAllImpl& c) = 0;
  virtual void visit(ComputeTileImpl& c) = 0;
};

class ExprImpl {
public:
  ExprImpl(){}
  virtual ~ExprImpl() {}
  virtual void visit(Visitor& visitor);
  virtual bool isVar()        {return false;}
  virtual bool isBinaryExpr() {return false;}
  virtual bool isForAll()     {return false;}
  virtual bool isDimension()  {return false;}
  virtual bool isUIntConst()  {return false;} 
  virtual bool isComputeTile() {return false;}
};

template<typename T>
class ConstImpl : public ExprImpl {
  T val_;
public:
  T val() const {return val_;}
  ConstImpl(T val) : val_(val) {}
};

class UIntConstImpl : public ConstImpl<uint> {
public:
  UIntConstImpl(uint v) : ConstImpl<uint>(v) {}
  virtual bool isUIntConst()  {return true;}
  virtual void visit(Visitor& visitor) {visitor.visit(*this);}
};

class VarImpl : public ExprImpl {
  std::string name_;
public:
  VarImpl(std::string name) : name_(name) {}
  virtual bool isVar()        {return true;}
  virtual void visit(Visitor& visitor) {visitor.visit(*this);}
};

class BinaryExprImpl : public ExprImpl {
  BinaryOp op_;
  std::shared_ptr<ExprImpl> op1_;
  std::shared_ptr<ExprImpl> op2_;

public:
  BinaryExprImpl(std::shared_ptr<ExprImpl> op1, BinaryOp op, std::shared_ptr<ExprImpl> op2) : 
    op1_(op1), op_(op), op2_(op2) {}
  std::shared_ptr<ExprImpl> op1() {return op1_;}
  std::shared_ptr<ExprImpl> op2() {return op2_;}
  BinaryOp op() {return op_;}
  virtual bool isBinaryExpr() {return true;}
  virtual void visit(Visitor& visitor) {visitor.visit(*this);}
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
  virtual bool isForAll()     {return true;}
  std::shared_ptr<DimensionImpl> var() {return var_;}
  std::shared_ptr<ExprImpl> baseExpr() {return baseExpr_;}
  uint lower() {return lower_;}
  uint upper() {return upper_;}
  virtual void visit(Visitor& visitor) {visitor.visit(*this);}
};

class ComputeTileImpl : public ExprImpl {
  std::vector<std::shared_ptr<ExprImpl>> dims_;

public:
  ComputeTileImpl(std::vector<std::shared_ptr<ExprImpl>> dims) : dims_(dims) {}
  std::vector<std::shared_ptr<ExprImpl>> dims() {return dims_;}
  virtual bool isComputeTile() {return true;}
  virtual void visit(Visitor& visitor) {visitor.visit(*this);}
  void genTileIndex(std::ostream& os, int indent, bool batched);
  std::shared_ptr<ComputeTileImpl> newTile(int dim, uint dimCoeff, uint dimAdder) {
    std::vector<std::shared_ptr<ExprImpl>> newDims;
    for (uint i = 0; i < dims_.size(); i++) {
      if (i == dim) {
        auto coeffConst = std::shared_ptr<UIntConstImpl>(new UIntConstImpl(dimCoeff));
        auto adderConst = std::shared_ptr<UIntConstImpl>(new UIntConstImpl(dimAdder));
        auto newDim = std::shared_ptr<ExprImpl>(new BinaryExprImpl(coeffConst, BinaryOp::Mul, dims_[i]));
        newDim = std::shared_ptr<ExprImpl>(new BinaryExprImpl(newDim, BinaryOp::Add, adderConst));
        newDims.push_back(newDim);
      } else {
        newDims.push_back(dims_[i]);
      }
    }
    return std::shared_ptr<ComputeTileImpl>(new ComputeTileImpl(newDims));
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
  virtual bool isDimension() {return true;}
  void genCondition(std::ostream& os) {
    os << "(" << lower() << "<=" << name() << " && " << name() << "<" << upper() << ")";
  }

  virtual void visit(Visitor& visitor) {visitor.visit(*this);}

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

std::pair<uint, uint> getTileAccessCoeff(std::shared_ptr<ExprImpl> dimExpr);