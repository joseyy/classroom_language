#include "../includes/ast.hpp"

ASTAssignment::
    ASTAssignment(std::string identifier,
                  ASTPointer expression) : identifier(identifier),
                                           expression(expression)
{
    // Evalaute the expression
    // Add the variable to the scope
    // std::cout << "ASTAssignment" << std::endl;
}

void ASTAssignment::print()
{
    std::cout << "ASTAssignment" << std::endl;
    std::cout << "identifier: " << identifier << std::endl;
    std::cout << "expression: " << std::endl;
    expression->print();
}

ASTBinaryOperation::
    ASTBinaryOperation(ASTPointer left_expression,
                       Token op,
                       ASTPointer right_expression) : left(left_expression),
                                                      op(op),
                                                      right(right_expression) {}

void ASTBinaryOperation::print()
{
    std::cout << "ASTBinaryOperation" << std::endl;
    std::cout << "left: " << std::endl;
    left->print();
    std::cout << "op: " << op.lexeme << std::endl;
    std::cout << "right: " << std::endl;
    right->print();
}

ASTVariableNode::
    ASTVariableNode(std::string identifier) : identifier(identifier)
{
}

void ASTVariableNode::print()
{
    std::cout << "ASTVariableNode" << std::endl;
    std::cout << "identifier: " << identifier << std::endl;
}

ASTNumberNode::
    ASTNumberNode(std::string value) : value_str(value)
{
}

void ASTNumberNode::print()
{
    std::cout << "ASTNumberNode" << std::endl;
    std::cout << "value: " << value_str << std::endl;
}

ASTFunctionCall::
    ASTFunctionCall(std::string identifier,
                    std::shared_ptr<std::vector<ASTPointer>> parameters) : identifier(identifier),
                                                                           parameters(parameters)
{
}

void ASTFunctionCall::print()
{
    std::cout << "ASTFunctionCall" << std::endl;
    std::cout << "identifier: " << identifier << std::endl;
    std::cout << "parameters: " << std::endl;
    for (auto &parameter : *parameters)
    {
        parameter->print();
    }
}

ASTBlock::
    ASTBlock(std::shared_ptr<std::vector<ASTPointer>> statements) : statements(statements)
{
}

void ASTBlock::print()
{
    std::cout << "ASTBlock" << std::endl;
    std::cout << "statements: " << std::endl;
    for (auto &statement : *statements)
    {
        statement->print();
    }
}

ASTFunctionDefinition::
    ASTFunctionDefinition(std::string identifier,
                          std::shared_ptr<std::vector<std::string>> parameters,
                          ASTPointer block) : identifier(identifier),
                                              parameters(parameters),
                                              block(block)
{
}

void ASTFunctionDefinition::print()
{
    std::cout << "ASTFunctionDefinition" << std::endl;
    std::cout << "identifier: " << identifier << std::endl;
    std::cout << "parameters: " << std::endl;
    for (auto &parameter : *parameters)
    {
        std::cout << parameter << std::endl;
    }
    std::cout << "block: " << std::endl;
    block->print();
}

ASTIfStatement::
    ASTIfStatement(ASTPointer condition,
                   ASTPointer block) : condition(condition),
                                       block(block)
{
}

void ASTIfStatement::print()
{
    std::cout << "ASTIfStatement" << std::endl;
    std::cout << "condition: " << std::endl;
    condition->print();
    std::cout << "block: " << std::endl;
    block->print();
}

ASTWhileStatement::
    ASTWhileStatement(ASTPointer condition,
                      ASTPointer block) : condition(condition),
                                          block(block)
{
}

void ASTWhileStatement::print()
{
    std::cout << "ASTWhileStatement" << std::endl;
    std::cout << "condition: " << std::endl;
    condition->print();
    std::cout << "block: " << std::endl;
    block->print();
}

ASTReturnStatement::
    ASTReturnStatement(ASTPointer expression) : expression(expression)
{
}

void ASTReturnStatement::print()
{
    std::cout << "ASTReturnStatement" << std::endl;
    std::cout << "expression: " << std::endl;
    expression->print();
}   