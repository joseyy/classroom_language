#ifndef AST_POINTER_HPP
#define AST_POINTER_HPP

#include "memory"
#include "./ast.hpp"

class ASTPointer
{
public:
    ASTPointer() : node(std::make_shared<ASTNode>(nullptr)) {}
    ASTPointer(std::shared_ptr<ASTNode> &node) : node(node) {}
    std::shared_ptr<ASTNode> operator->();
    std::shared_ptr<ASTNode> operator*();

private:
    std::shared_ptr<ASTNode> node;
};

#endif // AST_POINTER_HPP