#ifndef AST_HPP
#define AST_HPP

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "./token.hpp"
#include "./ast_pointer.hpp"
#include "./scope.hpp"

class ASTNode
{
public:
    virtual ~ASTNode() = default;
    virtual void print() = 0;
};

class ASTAssignment : public ASTNode
{
public:
    ASTAssignment(std::string identifier,
                  ASTPointer expression) : identifier(identifier),
                                           expression(expression) {}

    void print() override;

private:
    std::string identifier;
    ASTPointer expression;
};

class ASTBinaryOperation : public ASTNode
{
public:
    ASTBinaryOperation(ASTPointer left_expression,
                       Token op,
                       ASTPointer right_expression) : left(left_expression),
                                                      op(op),
                                                      right(right_expression) {}

    void print() override;

private:
    ASTPointer left;
    Token op;
    ASTPointer right;
};

class ASTVariableNode : public ASTNode
{
public:
    ASTVariableNode(std::string identifier) : identifier(identifier) {}

    void print() override;
    auto evaluate()
    {
    }

private:
    std::string identifier;
};

class ASTNumberNode : public ASTNode
{
public:
    ASTNumberNode(std::string value) : value_str(value) {}

    void print() override;

private:
    std::string value_str;
};

class ASTFunctionCall : public ASTNode
{
public:
    ASTFunctionCall(std::string identifier,
                    std::shared_ptr<std::vector<ASTPointer>> parameters) : identifier(identifier),
                                                                           parameters(parameters) {}

    ASTFunctionCall(std::string identifier,
                    ASTPointer caller) : identifier(identifier),
                                         caller(caller) {}

    void print() override;

private:
    std::string identifier;
    std::shared_ptr<std::vector<ASTPointer>> parameters;
    ASTPointer caller;
};

class ASTBlock : public ASTNode
{
public:
    ASTBlock(std::shared_ptr<std::vector<ASTPointer>> statements) : statements(statements) {}

    void print() override;

private:
    std::shared_ptr<std::vector<ASTPointer>> statements;
};

class ASTFunctionDefinition : public ASTNode
{
public:
    ASTFunctionDefinition(std::string identifier,
                          std::shared_ptr<std::vector<std::string>> parameters,
                          ASTPointer block) : identifier(identifier),
                                              parameters(parameters),
                                              block(block) {}

    void print() override;

private:
    std::string identifier;
    std::shared_ptr<std::vector<std::string>> parameters;
    ASTPointer block;
};

class ASTIfStatement : public ASTNode
{
public:
    ASTIfStatement(ASTPointer condition,
                   ASTPointer block) : condition(condition),
                                       block(block)
    {
    }

    void print() override;

private:
    ASTPointer condition;
    ASTPointer block;
};

class ASTWhileStatement : public ASTNode
{
public:
    ASTWhileStatement(ASTPointer condition,
                      ASTPointer block) : condition(condition),
                                          block(block)
    {
    }

    void print() override;

private:
    ASTPointer condition;
    ASTPointer block;
};

class ASTReturnStatement : public ASTNode
{
public:
    ASTReturnStatement(ASTPointer expression) : expression(expression) {}

    void print() override;

private:
    ASTPointer expression;
};

#endif // AST_HPP
