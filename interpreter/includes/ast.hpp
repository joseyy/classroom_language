#ifndef AST_HPP
#define AST_HPP

#include <memory>
#include <vector>
#include <string>
#include "./ast_pointer.hpp"
#include "./ast_visitor.hpp"

class ASTNode
{
public:
    virtual ~ASTNode() {}
    virtual void accept(const ASTVisitor &visitor) const = 0;
    virtual std::string type() const = 0;
};

class BinaryOpNode : public ASTNode
{
public:
    BinaryOpNode(std::unique_ptr<ASTPointer> left,
                 std::unique_ptr<ASTPointer> right,
                 const std::string &op)
        : left_(std::move(left)),
          right_(std::move(right)),
          op_(op) {}

    const ASTPointer left() const { return *left_; }
    const ASTPointer right() const { return *right_; }

    const std::string &op() const { return op_; }

    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "BinaryOpNode";
    }

private:
    std::unique_ptr<ASTPointer> left_;
    std::unique_ptr<ASTPointer> right_;
    std::string op_;
};

class NumberNode : public ASTNode
{
public:
    NumberNode(int value) : value_(value) {}
    int value() const { return value_; }
    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "NumberNode";
    }

private:
    int value_;
};

class AssignmentNode : public ASTNode
{
public:
    AssignmentNode(const std::string &target,
                   std::unique_ptr<ASTPointer> value)
        : target_(target), value_(std::move(value)) {}
    const std::string &target() const { return target_; }
    const ASTNode *value() const { return value_->get(); }
    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "AssignmentNode";
    }

private:
    std::string target_;
    std::unique_ptr<ASTPointer> value_;
};

class FunctionCallNode : public ASTNode
{
public:
    FunctionCallNode(const std::string &name, std::vector<std::unique_ptr<ASTPointer>> args)
        : name_(name), args_(std::move(args)) {}
    const std::string &name() const { return name_; }
    const std::vector<std::unique_ptr<ASTPointer>> &args() const { return args_; }
    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "FunctionCallNode";
    }

private:
    std::string name_;
    std::vector<std::unique_ptr<ASTPointer>> args_;
};

class MethodCallNode : public ASTNode
{
public:
    MethodCallNode(std::unique_ptr<ASTPointer> object,
                   const std::string &method_name,
                   std::vector<std::unique_ptr<ASTPointer>> args)
        : object_(std::move(object)), method_name_(method_name), args_(std::move(args)) {}
    const ASTNode *object() const { return object_->get(); }
    const std::string &method_name() const { return method_name_; }
    const std::vector<std::unique_ptr<ASTPointer>> &args() const { return args_; }
    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "MethodCallNode";
    }

private:
    std::unique_ptr<ASTPointer> object_;
    std::string method_name_;
    std::vector<std::unique_ptr<ASTPointer>> args_;
};

class VariableNode : public ASTNode
{
public:
    VariableNode(const std::string &name) : name_(name) {}

    const std::string &name() const { return name_; }

    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "VariableNode";
    }

private:
    std::string name_;
};

class IfNode : public ASTNode
{
public:
    IfNode(std::vector<std::unique_ptr<ASTPointer>> condition_expresion,
           std::vector<std::unique_ptr<ASTPointer>> body)
        : condition_expression_(std::move(condition_expresion)),
          body_(std::move(body))
    {
    }

    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "IfNode";
    }

private:
    std::vector<std::unique_ptr<ASTPointer>> condition_expression_;
    std::vector<std::unique_ptr<ASTPointer>> body_;
    ASTPointer else_body_;
    ASTPointer esle_if_condition_;
    ASTPointer else_if_body_;
};

class ForNode : public ASTNode
{
public:
    ForNode(std::vector<std::unique_ptr<ASTPointer>> condition_expresion,
            std::vector<std::unique_ptr<ASTPointer>> body)
        : condition_expression_(std::move(condition_expresion)),
          body_(std::move(body))
    {
    }

    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "ForNode";
    }   

private:
    std::vector<std::unique_ptr<ASTPointer>> condition_expression_;
    std::vector<std::unique_ptr<ASTPointer>> body_;
};

class EndOfLineNode : public ASTNode
{
public:
    EndOfLineNode() {}
    void accept(const ASTVisitor &visitor) const override
    {
        visitor.visit(*this);
    }

    std::string type() const override
    {
        return "EndOfLineNode";
    }
};

#endif // AST_HPP