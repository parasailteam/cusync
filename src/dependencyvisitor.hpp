#include "dependencyast.hpp"
#include "dependency.hpp"

#pragma once

class ComputeExprValue : public Visitor {
  std::stack<uint> valueStack;
  std::string dimName_;
  uint dimValue_;

public:
  ComputeExprValue(uint dimValue) : dimValue_(dimValue) {}
  uint computedValue() {return valueStack.top();}

  virtual void visit(BinaryExprImpl& expr) {
    expr.op1()->visit(*this);

    uint first = valueStack.top();
    valueStack.pop();

    expr.op2()->visit(*this);
    
    uint second = valueStack.top();
    valueStack.pop();

    if (expr.op() == BinaryOp::Add) {
      valueStack.push(first + second);
    } else if (expr.op() == BinaryOp::Mul) {
      valueStack.push(first * second);
    } else {assert(false);}
  }

  virtual void visit(DimensionImpl& expr) {
    valueStack.push(dimValue_);
  }

  virtual void visit(UIntConstImpl& c) {
    valueStack.push(c.val()); 
  }

  virtual void visit(ForAllImpl& c) {
    assert(false);
  }

  virtual void visit(ComputeTileImpl& c) {
    assert(false);
  }
};

class HasDimension : public Visitor {
  std::string dimName_;
  bool found_;

public:
  HasDimension(std::string dimName) : 
    dimName_(dimName), found_(false) {}
  bool found() {return found_;}
  
  virtual void visit(BinaryExprImpl& expr) {
    expr.op1()->visit(*this);
    expr.op2()->visit(*this);
  }
  virtual void visit(DimensionImpl& expr) {
    if (expr.name() == dimName_) {
      found_ = true;
    }
  }
  virtual void visit(UIntConstImpl& c) {}
  virtual void visit(ComputeTileImpl& c) {}
  virtual void visit(ForAllImpl& c) {}
};

class AllDimsInExpr : public Visitor {
  std::vector<DimensionImpl*> dims_;

public:
  AllDimsInExpr() {}
  std::vector<DimensionImpl*> getAllDims() {return dims_;}
  virtual void visit(BinaryExprImpl& expr) {
    expr.op1()->visit(*this);
    expr.op2()->visit(*this);
  }
  virtual void visit(DimensionImpl& expr) {
    if (std::find(dims_.begin(), dims_.end(), &expr) == dims_.end()) {
      dims_.push_back(&expr);
    }
  }
  virtual void visit(UIntConstImpl& c) {}
  virtual void visit(ComputeTileImpl& c) {}
  virtual void visit(ForAllImpl& c) {}
};

class ComputeBounds : public Visitor {
  std::stack<uint> valueStack;
  bool minOrMax; //True for min and False for max
  uint minValue_;
  uint maxValue_;
public:
  ComputeBounds() : minOrMax(false), minValue_(0), maxValue_(0) {}
  uint minValue() {return minValue_;}
  uint maxValue() {return maxValue_;}
  uint size() { return maxValue_ - minValue_;}
  
  void computeBounds(ExprImpl& expr) {
    minOrMax = true;
    expr.visit(*this);
    minValue_ = valueStack.top();
    valueStack.pop();
    minOrMax = false;
    expr.visit(*this);
    maxValue_ = valueStack.top();
    valueStack.pop();
  }

  virtual void visit(BinaryExprImpl& expr) {
    expr.op1()->visit(*this);
    uint first = valueStack.top();
    valueStack.pop();
    expr.op2()->visit(*this);
    uint second = valueStack.top();
    valueStack.pop();

    if (expr.op() == BinaryOp::Add) {
      valueStack.push(first + second);
    } else if (expr.op() == BinaryOp::Mul) {
      valueStack.push(first * second);
    } else {assert(false);}
  }

  virtual void visit(DimensionImpl& expr) {
    if (minOrMax) {
      //minimum
      valueStack.push(expr.lower());
    } else {
      //maximum
      valueStack.push(expr.upper());
    }
  }

  virtual void visit(UIntConstImpl& c) {
    valueStack.push(c.val()); 
  }

  virtual void visit(ForAllImpl& c) {
    assert(false);
  }

  virtual void visit(ComputeTileImpl& c) {}
};

class ComputeBoundsOfTile : public Visitor {
  std::stack<uint> valueStack;
  std::string dimName_;
  bool minOrMax; //True for min and False for max
  uint minValue_;
  uint maxValue_;
public:
  ComputeBoundsOfTile(std::string dimName) : 
    dimName_(dimName), minOrMax(false), minValue_(0), maxValue_(0) {}
  uint minValue() {return minValue_;}
  uint maxValue() {return maxValue_;}

  virtual void visit(BinaryExprImpl& expr) {
    expr.op1()->visit(*this);

    uint first = valueStack.top();
    valueStack.pop();

    expr.op2()->visit(*this);
    uint second = valueStack.top();
    valueStack.pop();

    if (expr.op() == BinaryOp::Add) {
      valueStack.push(first + second);
    } else if (expr.op() == BinaryOp::Mul) {
      valueStack.push(first * second);
    } else {assert(false);}
  }

  virtual void visit(DimensionImpl& expr) {
    if (expr.name() == dimName_) {
      if (minOrMax) {
        //minimum
        valueStack.push(expr.lower());
      } else {
        //maximum
        valueStack.push(expr.upper());
      }
    }
  }

  virtual void visit(UIntConstImpl& c) {
    valueStack.push(c.val()); 
  }

  virtual void visit(ForAllImpl& c) {
    if (c.var()->name() == dimName_) {
      if (minOrMax) {
        //minimum
        valueStack.push(c.lower());
      } else {
        //maximum
        valueStack.push(c.upper());
      }
    }
  }

  virtual void visit(ComputeTileImpl& c) {
    for (auto d : c.dims()) {
      HasDimension hasDim(dimName_);
      d->visit(hasDim);
      if (hasDim.found()) {
        minOrMax = true;
        d->visit(*this);
        minValue_ = valueStack.top();
        valueStack.pop();
        minOrMax = false;
        d->visit(*this);
        maxValue_ = valueStack.top();
        valueStack.pop();
        break;
      }
    }
  }
};