#ifndef PARSER_HPP
#define PARSER_HPP

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include "interpreter_error.hpp"
#include "lexer.hpp"
#include "token.hpp"
#include "ast.hpp"
#include "ast_pointer.hpp"

class Parser
{

public:
    Parser(std::vector<Token> tokens)
        : tokens_(std::move(tokens)), current_token_index_(0) {}

    ASTPointer parse_line()
    {
        ASTPointer statement = parse_statement();

        if (!match(TokenType::END_OF_LINE))
        {
            throw InterpreterError("Expected END_OF_LINE token");
        }
        consume(); // consume the END_OF_LINE token

        return statement;
    }

private:
    Token current_token() const
    {
        if (current_token_index_ != -1)
        {
            return tokens_[current_token_index_];
        }
        else
        {
            return Token{TokenType::END_OF_LINE, ""};
        }
    }
    bool match(TokenType type) const
    {
        return current_token().type == type;
    }
    void consume()
    {
        if (current_token_index_ < tokens_.size())
        {
            ++current_token_index_;
        }
        else
        {
            current_token_index_ = -1;
        }
    }

    ASTPointer parse_statement()
    {
        if (match(TokenType::PRINT))
        {
            return parse_print_statement();
        }
        else if (match(TokenType::IDENTIFIER))
        {
            return parse_assignment_statement();
        }
        else if (match(TokenType::FUNCTION))
        {
            return parse_function_definition();
        }
        else if (match(TokenType::IF))
        {
            return parse_control_flow_statement();
        }
        else if (match(TokenType::ELSEIF))
        {
            return parse_control_flow_statement();
        }
        else if (match(TokenType::ELSE))
        {
            return parse_control_flow_statement();
        }
        else if (match(TokenType::FOR))
        {
            return parse_control_flow_statement();
        }
        else if (match(TokenType::COLON))
        {
            return parse_control_flow_statement();
        }
        else if (match(TokenType::END))
        {
            return parse_control_flow_statement();
        }
        else
        {
            throw InterpreterError("Unexpected token");
        }
    }

    ASTPointer parse_print_statement();

    ASTPointer parse_assignment_statement()
    {
        std::string target = current_token().value;
        consume(); // consume the IDENTIFIER token
        if (match(TokenType::ASSIGNMENT))
        {
            consume(); // consume the ASSIGNMENT token
            ASTPointer value = parse_expression();
            return std::static_pointer_cast<ASTNode>(std::make_shared<AssignmentNode>(target, std::move(value)));
        }
        else if (match(TokenType::DOT))
        {
            consume(); // consume the DOT token
            return parse_method_call(std::make_unique<VariableNode>(target),
                                     current_token().value);
        }
        else
        {
            throw InterpreterError("Expected ASSIGNMENT or DOT token");
        }
    }

    ASTPointer parse_expression()
    {
        ASTPointer left = parse_term();

        while (match(TokenType::PLUS) ||
               match(TokenType::MINUS) ||
               match(TokenType::MUL) ||
               match(TokenType::DIV))
        {
            Token op = current_token();
            consume(); // consume the operator token
            ASTPointer right = parse_term();
            left = std::make_shared<BinaryOpNode>(std::move(left), std::move(right), op.value);
        }

        return left;
    }
    ASTPointer parse_term()
    {
        ASTPointer left = parse_factor();

        while (match(TokenType::PLUS) ||
               match(TokenType::MINUS) ||
               match(TokenType::MUL) ||
               match(TokenType::DIV))
        {
            Token op = current_token();
            consume(); // consume the operator token
            ASTPointer right = parse_factor();
            left = std::make_shared<BinaryOpNode>(std::move(left), std::move(right), op.value);
        }

        return left;
    }
    ASTPointer parse_factor()
    {
        if (match(TokenType::NUMBER))
        {
            int value = std::stoi(current_token().value);
            consume(); // consume the NUMBER token
            return std::static_pointer_cast<ASTNode>(std::make_shared<NumberNode>(value));
        }
        else if (match(TokenType::IDENTIFIER))
        {
            return parse_variable();
        }
        else if (match(TokenType::LPAREN))
        {
            consume(); // consume the LPAREN token
            ASTPointer expression = parse_expression();
            if (!match(TokenType::RPAREN))
            {
                throw InterpreterError("Expected RPAREN token");
            }
            consume(); // consume the RPAREN token
            return expression;
        }
        else if (match(TokenType::FUNCTION))
        {
            return parse_function_call(current_token().value);
        }
        else
        {
            throw InterpreterError("Unexpected token");
        }
    }
    // ASTPointer parse_assignment();
    ASTPointer parse_function_call(const std::string &name)
    {
        std::string name = current_token().value;
        consume(); // consume the FUNCTION token
        if (!match(TokenType::LPAREN))
        {
            throw InterpreterError("Expected LPAREN token");
        }
        consume(); // consume the LPAREN token
        std::vector<ASTPointer> args = parse_arguments();
        if (!match(TokenType::RPAREN))
        {
            throw InterpreterError("Expected RPAREN token");
        }
        consume(); // consume the RPAREN token
        return std::static_pointer_cast<ASTNode>(std::make_shared<FunctionCallNode>(name, std::move(args)));
    }

    ASTPointer parse_method_call(std::unique_ptr<ASTNode> object, const std::string &name);
    ASTPointer parse_variable();
    std::vector<ASTPointer> parse_arguments();
    std::vector<ASTPointer> parse_parameter_list();
    ASTPointer parse_function_definition();
    ASTPointer parse_control_flow_statement();
    std::vector<ASTPointer> parse_statement_list();
    std::vector<ASTPointer> parse_block();

    std::vector<Token> tokens_;
    size_t current_token_index_;

    {
        if (current_token_index_ < tokens_.size())
        {
            ++current_token_index_;
        }
        else
        {
            current_token_index_ = -1;
        }
    }
};

#endif // PARSER_HPP