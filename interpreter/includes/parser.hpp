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
    Parser()
        : current_token_index_(0) {}

    ASTPointer parse_line(std::vector<Token> token_line)
    {
        // copy token_line to tokens_
        tokens_.resize(token_line.size());
        std::copy(token_line.begin(), token_line.end(), std::begin(tokens_));

        ASTPointer statement = parse_statement();

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
        else if (match(TokenType::FOR))
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
            return ASTPointer(std::make_shared<AssignmentNode>(target, std::move(value)));
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

        if (match(TokenType::PLUS) ||
            match(TokenType::MINUS) ||
            match(TokenType::MUL) ||
            match(TokenType::DIV) ||
            match(TokenType::EQUAL) ||
            match(TokenType::NOT_EQUAL) ||
            match(TokenType::GREATER_THAN) ||
            match(TokenType::LESS_THAN) ||
            match(TokenType::GREATER_THAN_EQUAL) ||
            match(TokenType::LESS_THAN_EQUAL))
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

        if (match(TokenType::PLUS) ||
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
            return ASTPointer(std::make_shared<NumberNode>(value));
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
        return ASTPointer(std::make_shared<FunctionCallNode>(name, std::move(args)));
    }

    ASTPointer parse_method_call(std::unique_ptr<ASTNode> object,
                                 const std::string &name)
    {

        consume(); // consume token with name of method

        // Expect LPAREN token
        if (!match(TokenType::LPAREN))
        {
            throw InterpreterError("Expected LPAREN token");
        }
        consume(); // consume the LPAREN token

        std::vector<ASTPointer> args = parse_arguments();

        // Expect RPAREN token
        if (!match(TokenType::RPAREN))
        {
            throw InterpreterError("Expected RPAREN token");
        }
        consume(); // consume the RPAREN token
        return ASTPointer(std::make_shared<MethodCallNode>(std::move(object), name, std::move(args)));
    }

    std::vector<ASTPointer> parse_arguments()
    {
        std::vector<ASTPointer> args;
        if (!match(TokenType::RPAREN))
        {
            args = parse_parameter_list();
        }
        return args;
    }

    std::vector<ASTPointer> parse_parameter_list()
    {
        std::vector<ASTPointer> args;
        args.push_back(parse_expression());
        if (match(TokenType::COMMA))
        {
            consume(); // consume the COMMA token
            args.push_back(parse_expression());
        }
        return args;
    }

    ASTPointer parse_variable()
    {
        std::string name = current_token().value;
        consume(); // consume the IDENTIFIER token
        if (match(TokenType::DOT))
        {
            consume(); // consume the DOT token
            return parse_method_call(std::make_unique<VariableNode>(name),
                                     current_token().value);
        }
        else
        {
            return ASTPointer(std::make_shared<VariableNode>(name));
        }
    }

    ASTPointer parse_control_flow_statement()
    {
        if (match(TokenType::IF))
        {
            return parse_if_statement();
        }
        else if (match(TokenType::FOR))
        {
            return parse_for_statement();
        }
        else
        {
            throw InterpreterError("Unexpected token");
        }
    }
    ASTPointer parse_if_statement()
    {
        // save token value
        TokenType if_token = current_token().type;
        consume(); // consume the IF token
        ASTPointer blcok_condition_expression = parse_expression();
        if (!match(TokenType::COLON))
        {
            throw InterpreterError("Expected COLON token");
        }
        consume(); // consume the COLON token

        // start new block and get the lines tree
        std::vector<ASTPointer> block_expression = parse_block();

        return ASTPointer(std::make_shared<IfNode>(std::move(blcok_condition_expression),
                                                   std::move(block_expression)));
    }
    ASTPointer parse_for_statement()
    {
        std::string token_value = current_token().value;
        consume(); // consume the FOR token
        ASTPointer block_condition_expression = parse_expression();
        if (!match(TokenType::COLON))
        {
            throw InterpreterError("Expected COLON token");
        }
        consume(); // consume the COLON token

        // start new block and get the lines
        std::vector<ASTPointer> block_expression = parse_block();

        return ASTPointer(std::make_shared<ForNode>(std::move(block_condition_expression),
                                                    std::move(block_expression)));
    }

    std::vector<ASTPointer> parse_block()
    {

        Lexer lexer;
        Parser block_parser;
        std::vector<ASTPointer> block_lines;
        std::vector<Token> tokens;
        std::string input;

        // get last element of tokens
        while (tokens.back().type != TokenType::END_OF_LINE)
        {
            // get input for block
            std::cout << ">>    ";
            std::getline(std::cin, input);

            // Tokenize input
            tokens = lexer.tokenize(input);
            // Parse tokens into AST
            block_lines.push_back(block_parser.parse_line(tokens));
        }

        return block_lines;
    }

    ASTPointer parse_assignment();
    ASTPointer parse_function_definition();
    std::vector<ASTPointer> parse_statement_list();

    std::vector<Token> tokens_;
    size_t current_token_index_;
    bool in_block_ = false;
};

#endif // PARSER_HPP