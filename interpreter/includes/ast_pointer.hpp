#ifndef AST_POINTER_HPP
#define AST_POINTER_HPP

#include <memory>
#include "ast.hpp"

class ASTPointer
{
public:
    ASTPointer() : ptr(nullptr) {}
    ASTPointer(std::shared_ptr<ASTNode> ptr) : ptr(ptr) {}

    ASTNode *get() const
    {
        return ptr.get();
    }

    ASTNode &operator*() const
    {
        return *ptr;
    }

    ASTNode *operator->() const
    {
        return ptr.get();
    }

    operator bool() const
    {
        return ptr != nullptr;
    }

    ASTPointer &operator=(std::shared_ptr<ASTNode> ptr)
    {
        this->ptr = ptr;
        return *this;
    }

private:
    std::shared_ptr<ASTNode> ptr;
};

#endif // AST_POINTER_HPP