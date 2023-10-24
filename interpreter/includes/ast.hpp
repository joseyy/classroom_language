#ifndef AST_HPP
#define AST_HPP

#include <memory>
#include <vector>
#include <string>

class ASTNode
{
public:
    virtual ~ASTNode() {}
    virtual void accept(class ASTVisitor &visitor) const = 0;
};

class BinaryOpNode : public ASTNode
{
public:
    BinaryOpNode(std::unique_ptr<ASTNode> left,
                 std::unique_ptr<ASTNode> right,
                 const std::string &op)
        : left_(std::move(left)),
          right_(std::move(right)),
          op_(op) {}

    const ASTNode *left() const { return left_.get(); }
    const ASTNode *right() const { return right_.get(); }

    const std::string &op() const { return op_; }

    void accept(ASTVisitor &visitor) const override;

private:
    std::unique_ptr<ASTNode> left_;
    std::unique_ptr<ASTNode> right_;
    std::string op_;
};

class NumberNode : public ASTNode
{
public:
    NumberNode(int value) : value_(value) {}
    int value() const { return value_; }
    void accept(ASTVisitor &visitor) const override;

private:
    int value_;
};

class AssignmentNode : public ASTNode
{
public:
    AssignmentNode(const std::string &target,
                   std::unique_ptr<ASTNode> value)
        : target_(target), value_(std::move(value)) {}
    const std::string &target() const { return target_; }
    const ASTNode *value() const { return value_.get(); }
    void accept(ASTVisitor &visitor) const override;

private:
    std::string target_;
    std::unique_ptr<ASTNode> value_;
};

class FunctionCallNode : public ASTNode
{
public:
    FunctionCallNode(const std::string &name, std::vector<std::unique_ptr<ASTNode>> args)
        : name_(name), args_(std::move(args)) {}
    const std::string &name() const { return name_; }
    const std::vector<std::unique_ptr<ASTNode>> &args() const { return args_; }
    void accept(ASTVisitor &visitor) const override;

private:
    std::string name_;
    std::vector<std::unique_ptr<ASTNode>> args_;
};

class MethodCallNode : public ASTNode
{
public:
    MethodCallNode(std::unique_ptr<ASTNode> object, const std::string &method_name, std::vector<std::unique_ptr<ASTNode>> args)
        : object_(std::move(object)), method_name_(method_name), args_(std::move(args)) {}
    const ASTNode *object() const { return object_.get(); }
    const std::string &method_name() const { return method_name_; }
    const std::vector<std::unique_ptr<ASTNode>> &args() const { return args_; }
    void accept(ASTVisitor &visitor) const override;

private:
    std::unique_ptr<ASTNode> object_;
    std::string method_name_;
    std::vector<std::unique_ptr<ASTNode>> args_;
};

class VariableNode : public ASTNode
{
public:
    VariableNode(const std::string &name) : name_(name) {}

    const std::string &name() const { return name_; }

private:
    std::string name_;
};

class ASTVisitor
{
public:
    virtual ~ASTVisitor() {}
    virtual void visit(const BinaryOpNode &node) = 0;
    virtual void visit(const NumberNode &node) = 0;
    virtual void visit(const AssignmentNode &node) = 0;
    virtual void visit(const FunctionCallNode &node) = 0;
    virtual void visit(const MethodCallNode &node) = 0;
    virtual void visit(const VariableNode &node) = 0;
};

#endif // AST_HPP