#include "dependency.hpp"

class BinaryExpr;
class Dimension;

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

template<typename T>
class Const : public Expr {
public:
  Const(T val) : Expr(std::make_shared<ConstImpl<T>>(val)) 
  {}
};

using UIntConst = Const<uint>;

class Var : public Expr {
public:
  Var(std::string name) : Expr(std::make_shared<VarImpl>(new VarImpl(name))) {}
};

class BinaryExpr : public Expr {
public:
  enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div
  };

  BinaryExpr(Expr op1, BinaryOp op, Expr op2) : Expr(std::make_shared<BinaryExprImpl>(new BinaryExprImpl(op1.impl(), op, op2.impl()))) {}
};


class ForAll : public Expr {
public:
  ForAll(Dimension var, Expr baseExpr, uint lower, uint upper) : 
    Expr(std::make_shared<ForAllImpl>(new ForAllImpl(var.impl(), baseExpr.impl(), lower, upper)))
   {}
};

BinaryExpr Expr::operator*(Expr op2) {
  return BinaryExpr(*this, BinaryExpr::Mul, op2.impl());
}

BinaryExpr Expr::operator*(uint op2) {
  return BinaryExpr(*this, BinaryExpr::Mul, UIntConst(op2));
}

BinaryExpr operator*(uint op1, Expr& op2) {
  return BinaryExpr(op2, BinaryExpr::Mul, UIntConst(op1));
}

BinaryExpr Expr::operator+(uint op2) {
  return BinaryExpr(*this, BinaryExpr::Mul, UIntConst(op2));
}

BinaryExpr Expr::operator+(Expr op2) {
  return BinaryExpr(*this, BinaryExpr::Mul, op2.impl());
}


class ComputeTile : public Expr {
public:
  ComputeTile(std::vector<Expr> dims) : Expr(nullptr) {
    std::vector<std::shared_ptr<ExprImpl>> dimImpls;
    std::transform(dims.begin(), dims.end(), dimImpls.begin(), [](Expr& e) {return e.impl();});
    impl_ = std::make_shared<ExprImpl>(new ComputeTileImpl(dimImpls));
  }

  std::shared_ptr<ComputeTileImpl> impl() { return std::dynamic_pointer_cast<ComputeTileImpl>(impl_);}
};

//Specify dependency by adding all src tiles for a dst tile
class Dependency {
  std::vector<std::shared_ptr<ExprImpl>> srcTiles_;
  std::shared_ptr<ComputeTileImpl> dstTile_;

public:
  Dependency(std::vector<Expr> srcTiles, ComputeTile dstTile) : 
    dstTile_(dstTile.impl()) {
    std::transform(srcTiles.begin(), srcTiles.end(), srcTiles_, [](Expr& e){return e.impl();});
  }

  Dependency(Expr srcTile, ComputeTile dstTile) : 
    srcTiles_({srcTile.impl()}), dstTile_(dstTile.impl()) {}
};


class Dimension : public Expr {
public:
  Dimension(std::string name, uint lower, uint upper) : 
    Expr(std::make_shared<DimensionImpl>(name, lower, upper)) 
  {
      assert(upper > lower);
  }

  std::shared_ptr<DimensionImpl> impl() {return std::dynamic_pointer_cast<DimensionImpl>(impl_);}
};