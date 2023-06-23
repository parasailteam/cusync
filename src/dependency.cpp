#include "dependency.hpp"
#include "dependencyvisitor.hpp"
#include "utils.h"

const std::string TILE = "tile.";

void ExprImpl::visit(Visitor& visitor) {visitor.visit(*this);}

void Visitor::visit(ExprImpl& expr){assert(false); expr.visit(*this);};

BinaryExpr Expr::operator*(Expr op2) {
  return BinaryExpr(*this, BinaryOp::Mul, op2.impl());
}

BinaryExpr Expr::operator*(uint op2) {
  return BinaryExpr(*this, BinaryOp::Mul, UIntConst(op2));
}

BinaryExpr operator*(uint op1, Expr& op2) {
  return BinaryExpr(op2, BinaryOp::Mul, UIntConst(op1));
}

BinaryExpr Expr::operator+(uint op2) {
  return BinaryExpr(*this, BinaryOp::Add, UIntConst(op2));
}

BinaryExpr Expr::operator+(Expr op2) {
  return BinaryExpr(*this, BinaryOp::Add, op2.impl());
}

void ComputeTileImpl::genTileIndex(std::ostream& os, int indent, bool batched) {
  if (batched) {
    os << indentStr(indent) << "if (";
    for (auto iter = dims_.begin(); iter != dims_.end();) {
      auto dim = *iter;
      AllDimsInExpr getAllDims;
      dim->visit(getAllDims);
      auto allDims = getAllDims.getAllDims();
      auto coeffAdder = getTileAccessCoeff(dim);
      assert(allDims.size() == 1);
      bool addCond = true;
      if (coeffAdder.first == 1 and coeffAdder.second == 0) {
        // os << "true";
        addCond = false;
      } else {
        os << TILE << allDims[0]->name();
        if (coeffAdder.first != 1)
          os << "%" << coeffAdder.first;
        os << " <= " << coeffAdder.second;
      }
      iter++;
      if (addCond && iter != dims_.end()) {
        os << " && ";
      }
    }
    os << ")";
  }

  os << indentStr(indent) << "return ";

  for (auto iter = dims_.begin(); iter != dims_.end();) {
    auto dim = *iter;
    AllDimsInExpr getAllDims;
    dim->visit(getAllDims);
    auto allDims = getAllDims.getAllDims();
    auto coeffAdder = getTileAccessCoeff(dim);
    assert(allDims.size() == 1);
    os << "(" + TILE + allDims[0]->name();
    if (batched && coeffAdder.second != 0) {
      os << " - " << coeffAdder.second;
    }
    os << ")";
    if (batched && coeffAdder.first != 1) {
      os << "/" << coeffAdder.first;
    }
    if (iter != dims_.begin()) {
      os << "*";
      os << "(";
      for (auto iter2 = dims_.begin(); iter2 != iter;) {
        ComputeBounds getBounds;
        getBounds.computeBounds(*(*iter2));
        os << getBounds.size();
        iter2++;
        if (iter2 != iter) {
          os << "*";
        }
      }
      os << ")";
    }
    iter++;
    if (iter != dims_.end()) {
      os << " + ";
    }
  }

  os << ";";
}

void ComputeTileImpl::genSchedIndex(std::ostream& os, int indent, int tileIndex, std::string yt) {
  os << indentStr(indent) << "if (";
  for (auto iter = dims_.begin(); iter != dims_.end();) {
    auto dim = *iter;
    AllDimsInExpr getAllDims;
    dim->visit(getAllDims);
    auto allDims = getAllDims.getAllDims();
    auto coeffAdder = getTileAccessCoeff(dim);
    assert(allDims.size() == 1);
    if (coeffAdder.first == 1 and coeffAdder.second == 0) {
      os << "true";
    } else {
      os << TILE << allDims[0]->name();
      if (coeffAdder.first != 1)
        os << "%" << coeffAdder.first;
      os << " <= " << coeffAdder.second;
    }
    iter++;
    if (iter != dims_.end()) {
      os << " && ";
    }
  }
  os << ")";

  os << indentStr(indent + 1) << "yt = " << tileIndex << ";";
}

std::pair<uint, uint> getTileAccessCoeff(std::shared_ptr<ExprImpl> dimExpr) {
  ComputeExprValue visitor0(0);
  dimExpr->visit(visitor0);
  uint adder = visitor0.computedValue();

  ComputeExprValue visitor1(1);
  dimExpr->visit(visitor1);
  uint v = visitor1.computedValue();
  uint coeff = v - adder;

  return std::make_pair(coeff, adder);
}

void DimensionImpl::genCondition(std::ostream& os) {
  os << "(" << lower() << "<=" << TILE << name() << " && " << TILE << name() << "<" << upper() << ")";
}