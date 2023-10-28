#ifndef AST_VISITOR_HPP
#define AST_VISITOR_HPP

#include <iostream>
#include "./ast.hpp"

class BinaryOpNode;
class NumberNode;
class AssignmentNode;
class FunctionCallNode;
class MethodCallNode;
class VariableNode;
class IfNode;
class ForNode;
class EndOfLineNode;

class ASTVisitor
{
public:
    virtual ~ASTVisitor() {}
    virtual void visit(const ASTNode &node) const = 0;
};

class BinaryOpNodeVisitor : public ASTVisitor
{
public:
    void visit(const ASTNode &node) const override
    {
        std::cout << "Visiting BinaryOpNode" << std::endl;
        // Visit the left and right operands of the binary operation
        // PRINT ROOT NODE
        std::cout << node.op() << std::endl;
        node.left()->accept(*this);
        node.right()->accept(*this);
        // Perform any additional processing for the binary operation
        // ...
    }
};

class NumberNodeVisitor : public ASTVisitor
{
public:
    void visit(const NumberNode &node) const override
    {
        std::cout << "Visiting NumberNode" << std::endl;
        // Visit the number value
        // ...
    }
};

class AssignmentNodeVisitor : public ASTVisitor
{
public:
    void visit(const AssignmentNode &node) const override
    {
        std::cout << "Visiting AssignmentNode" << std::endl;
        // Visit the target variable
        // ...
        // Visit the value being assigned
        // ...
    }
};

class FunctionCallNodeVisitor : public ASTVisitor
{
public:
    void visit(const FunctionCallNode &node) const override
    {
        std::cout << "Visiting FunctionCallNode" << std::endl;
        // Visit the function name
        // ...
        // Visit the function arguments
        // ...
    }
};

class MethodCallNodeVisitor : public ASTVisitor
{
public:
    void visit(const MethodCallNode &node) const override
    {
        std::cout << "Visiting MethodCallNode" << std::endl;
        // Visit the object
        // ...
        // Visit the method name
        // ...
        // Visit the method arguments
        // ...
    }
};

class VariableNodeVisitor : public ASTVisitor
{
public:
    void visit(const VariableNode &node) const override
    {
        std::cout << "Visiting VariableNode" << std::endl;
        // Visit the variable name
        // ...
    }
};

class IfNodeVisitor : public ASTVisitor
{
public:
    void visit(const IfNode &node) const override
    {
        std::cout << "Visiting IfNode" << std::endl;
        // Visit the condition expression
        // ...
        // Visit the body
        // ...
        // Visit the else body
        // ...
        // Visit the else if condition
        // ...
        // Visit the else if body
        // ...
    }
};

class ForNodeVisitor : public ASTVisitor
{
public:
    void visit(const ForNode &node) const override
    {
        std::cout << "Visiting ForNode" << std::endl;
        // Visit the condition expression
        // ...
        // Visit the body
        // ...
    }
};

class EndOfLineNodeVisitor : public ASTVisitor
{
public:
    void visit(const EndOfLineNode &node) const override
    {
        std::cout << "Visiting EndOfLineNode" << std::endl;
        // Visit the end of line
        // ...
    }
};

#endif // AST_VISITOR_HPP