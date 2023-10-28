
#include "../includes/ast_pointer.hpp"

ASTPointer::ASTPointer() : node(std::make_shared<ASTNode>(nullptr)) {}

ASTPointer::ASTPointer(std::shared_ptr<ASTNode> &node) : node(node) {}

std::shared_ptr<ASTNode> ASTPointer::operator*()
{
    return node;
}

std::shared_ptr<ASTNode> ASTPointer::operator->()
{
    return node;
}

