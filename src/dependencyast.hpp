#include <memory>

#include "dependency.hpp"

#pragma once

class Expr;
class BinaryExpr;
class Dimension;
class ForAll;
class ComputeTile;

class Expr {
protected:
  std::shared_ptr<ExprImpl> impl_;
public:
  std::shared_ptr<ExprImpl> impl() {return impl_;}
  BinaryExpr operator*(Expr op2);
  BinaryExpr operator*(uint op2);
  BinaryExpr operator+(Expr op2);
  BinaryExpr operator+(uint op2);

protected:
  Expr(std::shared_ptr<ExprImpl> impl) : impl_(impl) {}
};

template<typename T, typename Impl>
class Const : public Expr {
public:
  Const(T val) : Expr(std::shared_ptr<Impl>(new Impl(val))) 
  {}
};

using UIntConst = Const<uint, UIntConstImpl>;

class Var : public Expr {
public:
  Var(std::string name) : Expr(std::shared_ptr<VarImpl>(new VarImpl(name))) {}
};

class BinaryExpr : public Expr {
public:
  BinaryExpr(Expr op1, BinaryOp op, Expr op2) : Expr(std::shared_ptr<BinaryExprImpl>(new BinaryExprImpl(op1.impl(), op, op2.impl()))) {}
};

class Dimension : public Expr {
public:
  Dimension(std::string name, uint lower, uint upper) : 
    Expr(std::shared_ptr<DimensionImpl>(new DimensionImpl(name, lower, upper))) 
  {
      assert(upper > lower);
  }

  std::shared_ptr<DimensionImpl> impl() {
    return std::dynamic_pointer_cast<DimensionImpl>(impl_);
  }
};

class ForAll : public Expr {
public:
  ForAll(Dimension var, Expr baseExpr, uint lower, uint upper) : 
    Expr(std::shared_ptr<ForAllImpl>(new ForAllImpl(var.impl(), baseExpr.impl(), lower, upper)))
   {}
};

BinaryExpr operator*(uint op1, Expr& op2);

class ComputeTile : public Expr {
public:
  ComputeTile(std::vector<Expr> dims) : Expr(nullptr) {
    std::vector<std::shared_ptr<ExprImpl>> dimImpls(dims.size());
    std::transform(dims.begin(), dims.end(), dimImpls.begin(), [](Expr& e) {return e.impl();});
    impl_ = std::shared_ptr<ComputeTileImpl>(new ComputeTileImpl(dimImpls));
  }

  std::shared_ptr<ComputeTileImpl> impl() { return std::dynamic_pointer_cast<ComputeTileImpl>(impl_);}
};

class GridDim {
  std::shared_ptr<DimensionImpl> x_;
  std::shared_ptr<DimensionImpl> y_;
  std::shared_ptr<DimensionImpl> z_;

public:
  GridDim(Dimension x, Dimension y) : x_(x.impl()), y_(y.impl()) {}
  
  std::shared_ptr<DimensionImpl> x() {return x_;}
  std::shared_ptr<DimensionImpl> y() {return y_;}
  std::shared_ptr<DimensionImpl> z() {return z_;}

  uint numDims() {return 2;}
  std::shared_ptr<DimensionImpl> dim(uint i) {
    assert (i < 3);
    if (i == 0) return x();
    if (i == 1) return y();
    return z();
  }
};


//Specify dependency by adding all src tiles for a dst tile
class Dependency {
  std::vector<std::shared_ptr<ExprImpl>> srcTiles_;
  std::shared_ptr<ComputeTileImpl> dstTile_;
  GridDim grid1_; 
  GridDim grid2_;
public:
  Dependency(GridDim grid1, GridDim grid2, std::vector<Expr> srcTiles, ComputeTile dstTile) : 
    dstTile_(dstTile.impl()), srcTiles_(srcTiles.size()), grid1_(grid1), grid2_(grid2) {
    std::transform(srcTiles.begin(), srcTiles.end(), srcTiles_.begin(), [](Expr& e){return e.impl();});
  }

  Dependency(GridDim grid1, GridDim grid2, Expr srcTile, ComputeTile dstTile) : 
    srcTiles_({srcTile.impl()}), dstTile_(dstTile.impl()), grid1_(grid1), grid2_(grid2) {}
  
  std::vector<std::shared_ptr<ExprImpl>> srcTiles() {return srcTiles_;}
  std::shared_ptr<ComputeTileImpl> dstTile() {return dstTile_;}
  GridDim grid1() {return grid1_;}
  GridDim grid2() {return grid2_;}
};