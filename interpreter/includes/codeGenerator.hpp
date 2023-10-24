#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include <iostream>
#include <sstream>
#include "ast.hpp"

class CodeGenerator : public ASTVisitor
{
public:
    CodeGenerator() : out_() {}
    std::string generate(const ASTNode &node)
    {
        out_.str("");
        node.accept(*this);
        return out_.str();
    }

    void visit(const BinaryOpNode &node) override
    {
        out_ << "(";
        node.left()->accept(*this);
        out_ << " " << node.op() << " ";
        node.right()->accept(*this);
        out_ << ")";
    }
    void visit(const NumberNode &node) override
    {
        out_ << node.value();
    }
    void visit(const AssignmentNode &node) override
    {
        out_ << node.target() << " = ";
        node.value()->accept(*this);
        out_ << ";";
    }
    void visit(const FunctionCallNode &node) override
    {
        out_ << node.name() << "(";
        for (size_t i = 0; i < node.args().size(); i++)
        {
            if (i > 0)
            {
                out_ << ", ";
            }
            node.args()[i]->accept(*this);
        }
        out_ << ")";
    }
    void visit(const MethodCallNode &node) override
    {
        node.object()->accept(*this);
        out_ << "." << node.method_name() << "(";
        for (size_t i = 0; i < node.args().size(); i++)
        {
            if (i > 0)
            {
                out_ << ", ";
            }
            node.args()[i]->accept(*this);
        }
        out_ << ")";
    }

private:
    std::ostringstream out_;
};

#endif // CODE_GENERATOR_HPP