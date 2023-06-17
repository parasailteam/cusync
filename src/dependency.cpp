#include "dependency.hpp"
#include "dependencyvisitor.hpp"

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
