#ifndef PARSER_HPP
#define PARSER_HPP

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include "./interpreter_error.hpp"
#include "./lexer.hpp"
#include "./token.hpp"
#include "./ast.hpp"
#include "./ast_pointer.hpp"

bool DEBUG = false;

class Parser
{

public:
    Parser()
        : current_token_index_(0)
    {
        if (DEBUG)
            std::cout << "Parser constructor" << std::endl;
        numBlocks_++;
    }

    ASTPointer parse_line(std::vector<Token> token_line)
    {
        current_token_index_ = 0;
        if (DEBUG)
            std::cout << "parse_line" << std::endl;

        tokens_ = token_line;
        ASTPointer statement = parse_statement();

        return statement;
    }

    ~Parser()
    {
        if (DEBUG)
            std::cout << "Parser destructor" << std::endl;
        numBlocks_--;
    }

private:
    Token current_token() const
    {
        if (current_token_index_ < tokens_.size())
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
    }

    ASTPointer parse_statement()
    {
        if (DEBUG)
            std::cout << "parse_statement" << std::endl;

        if (match(TokenType::PRINT))
        {
            // return parse_print_statement();
        }
        else if (match(TokenType::IDENTIFIER))
        {
            return parse_assignment_statement();
        }
        else if (match(TokenType::IF))
        {
            return parse_control_flow_statement();
        }
        else if (match(TokenType::FOR))
        {
            return parse_control_flow_statement();
        }
        else if (match(TokenType::NUMBER))
        {
            return parse_expression();
        }
        else
        {
            std::cout << "current_token().value: " << current_token().value << std::endl;
            throw InterpreterError("Unexpected token");
        }
    }

    ASTPointer parse_print_statement();

    ASTPointer parse_assignment_statement()
    {
        if (DEBUG)
            std::cout << "parse_assignment_statement" << std::endl;

        std::string target = current_token().value;
        consume(); // consume the IDENTIFIER token
        if (match(TokenType::ASSIGNMENT))
        {
            consume(); // consume the ASSIGNMENT token
            auto value = parse_expression();
            return ASTPointer(std::make_shared<AssignmentNode>(target,
                                                               std::move(std::make_unique<ASTPointer>(value))));
        }
        else if (match(TokenType::DOT))
        {
            consume(); // consume the DOT token
            return parse_method_call(ASTPointer(std::make_shared<VariableNode>(target)),
                                     current_token().value);
        }
        else if (match(TokenType::LPAREN))
        {
            consume(); // consume the LPAREN token
            return parse_function_call(target);
        }
        else
        {
            throw InterpreterError("Expected ASSIGNMENT, DOT or LPAREN tokens");
        }
    }

    ASTPointer parse_expression()
    {
        if (DEBUG)
            std::cout << "parse_expression" << std::endl;

        ASTPointer left = parse_term();

        while (match(TokenType::PLUS) ||
               match(TokenType::MINUS) ||
               match(TokenType::MUL) ||
               match(TokenType::DIV) ||
               match(TokenType::EQUAL_TO) ||
               match(TokenType::NOT_EQUAL_TO) ||
               match(TokenType::GREATER_THAN) ||
               match(TokenType::LESS_THAN) ||
               match(TokenType::GREATER_THAN_EQUAL_TO) ||
               match(TokenType::LESS_THAN_EQUAL_TO))
        {
            Token op = current_token();
            consume(); // consume the operator token
            if (!(match(TokenType::IDENTIFIER) ||
                  match(TokenType::NUMBER) ||
                  match(TokenType::LPAREN)))
                throw InterpreterError("Expected expression after operator");

            ASTPointer right = parse_term();
            left = std::make_shared<BinaryOpNode>(std::move(std::make_unique<ASTPointer>(left)),
                                                  std::move(std::make_unique<ASTPointer>(right)),
                                                  op.value);
        }

        return left;
    }
    ASTPointer parse_term()
    {
        if (DEBUG)
            std::cout << "parse_term" << std::endl;

        ASTPointer left = parse_factor();

        if (DEBUG)
            std::cout << "parse_term: token value " << current_token().value << std::endl;

        while (match(TokenType::PLUS) ||
               match(TokenType::MINUS) ||
               match(TokenType::MUL) ||
               match(TokenType::DIV))
        {
            Token op = current_token();
            consume(); // consume the operator token
            if (!(match(TokenType::IDENTIFIER) ||
                  match(TokenType::NUMBER) ||
                  match(TokenType::LPAREN)))
                throw InterpreterError("Expected expression after operator");
            ASTPointer right = parse_factor();
            left = std::make_shared<BinaryOpNode>(std::move(std::make_unique<ASTPointer>(left)),
                                                  std::move(std::make_unique<ASTPointer>(right)),
                                                  op.value);
        }
        return left;
    }
    ASTPointer parse_factor()
    {

        if (DEBUG)
        {
            std::cout << "parse_factor: token value " << current_token().value << std::endl;
        }

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
        else if (match(TokenType::END_OF_LINE))
        {
            return ASTPointer(std::make_shared<EndOfLineNode>());
        }
        else
        {
            throw InterpreterError("Parse Factor: Unexpected token");
        }
    }

    ASTPointer parse_function_call(const std::string &name)
    {
        if (DEBUG)
            std::cout << "parse_function_call" << std::endl;

        auto args = parse_arguments();
        if (!match(TokenType::RPAREN))
        {
            throw InterpreterError("Parse_function_call: Expected RPAREN token");
        }
        consume(); // consume the RPAREN token
        return ASTPointer(std::make_shared<FunctionCallNode>(name,
                                                             std::move(*args)));
    }

    ASTPointer parse_method_call(ASTPointer object,
                                 const std::string &name)
    {
        if (DEBUG)
            std::cout << "parse_method_call" << std::endl;

        consume(); // consume token with name of method

        // Expect LPAREN token
        if (!match(TokenType::LPAREN))
        {
            throw InterpreterError("Expected LPAREN token");
        }
        consume(); // consume the LPAREN token

        auto args = parse_arguments();

        // Expect RPAREN token
        if (!match(TokenType::RPAREN))
        {
            throw InterpreterError("Expected RPAREN token");
        }
        consume(); // consume the RPAREN token
        return ASTPointer(std::make_shared<MethodCallNode>(std::move(std::make_unique<ASTPointer>(object)),
                                                           name, std::move(*args)));
    }

    std::shared_ptr<std::vector<std::unique_ptr<ASTPointer>>>
    parse_arguments()
    {
        if (DEBUG)
            std::cout << "parse_arguments" << std::endl;

        auto args = std::make_shared<std::vector<std::unique_ptr<ASTPointer>>>();
        if (!match(TokenType::RPAREN))
        {
            args = parse_parameter_list(args);
        }
        return args;
    }

    std::shared_ptr<std::vector<std::unique_ptr<ASTPointer>>>
    parse_parameter_list(
        std::shared_ptr<std::vector<std::unique_ptr<ASTPointer>>> args)
    {
        if (DEBUG)
            std::cout << "parse_parameter_list" << std::endl;

        args->push_back(std::make_unique<ASTPointer>(parse_expression()));
        while (match(TokenType::COMMA))
        {
            consume(); // consume the COMMA token
            if (DEBUG)
                std::cout << "parse_parameter_list while " << current_token().value << std::endl;
            if (!(match(TokenType::IDENTIFIER) ||
                  match(TokenType::NUMBER)))
                throw InterpreterError("Expected IDENTIFIER or Number token after COMMA");

            args->push_back(std::make_unique<ASTPointer>(parse_expression()));
        }
        return args;
    }

    ASTPointer parse_variable()
    {
        if (DEBUG)
            std::cout << "parse_variable" << std::endl;

        std::string name = current_token().value;
        consume(); // consume the IDENTIFIER token
        if (match(TokenType::DOT))
        {
            consume(); // consume the DOT token
            return parse_method_call(ASTPointer(std::make_shared<VariableNode>(name)),
                                     current_token().value);
        }
        else if (match(TokenType::LPAREN))
        {

            consume(); // consume the LPAREN token
            return parse_function_call(name);
        }
        else
        {
            return ASTPointer(std::make_shared<VariableNode>(name));
        }
    }

    ASTPointer parse_control_flow_statement()
    {
        if (DEBUG)
            std::cout << "parse_control_flow_statement" << std::endl;

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
            throw InterpreterError("Parse_Control_FLow_Statement: Unexpected token");
        }
    }
    ASTPointer parse_if_statement()
    {
        if (DEBUG)
            std::cout << "parse_if_statement" << std::endl;

        consume(); // consume the IF token
        auto block_condition_expression = parse_arguments();
        if (!match(TokenType::COLON))
        {
            throw InterpreterError("Expected COLON token");
        }
        consume(); // consume the COLON token

        // start new block and get the lines tree
        auto body = parse_block();

        return ASTPointer(std::make_shared<IfNode>(std::move(*block_condition_expression),
                                                   std::move(*body)));
    }
    ASTPointer parse_for_statement()
    {
        if (DEBUG)
            std::cout << "parse_for_statement" << std::endl;

        std::string token_value = current_token().value;
        consume(); // consume the FOR token
        auto block_condition_expression = parse_arguments();
        if (!match(TokenType::COLON))
        {
            std::cout << "current_token().value: " << current_token().value << std::endl;
            throw InterpreterError("Expected COLON token");
        }
        consume(); // consume the COLON token

        // start new block and get the lines
        auto block_expression = parse_block();

        return ASTPointer(std::make_shared<ForNode>(std::move(*block_condition_expression),
                                                    std::move(*block_expression)));
    }

    std::shared_ptr<std::vector<std::unique_ptr<ASTPointer>>>
    parse_block()
    {
        if (DEBUG)
            std::cout << "parse_block" << std::endl;

        Lexer lexer;
        Parser block_parser;
        auto block_lines = std::make_shared<std::vector<std::unique_ptr<ASTPointer>>>();
        std::vector<Token> tokens;
        std::string input;

        if (DEBUG)
            std::cout << "parse_block: after initialization" << std::endl;

        // get input for block
        std::cout << ">>";
        for (int i = 0; i < numBlocks_; i++)
        {
            std::cout << "  ";
        }
        std::getline(std::cin, input);

        // Tokenize input
        tokens = lexer.tokenize(input);

        // get last element of tokens
        while (input != "end")
        {
            // Parse tokens into AST
            if (!block_lines && DEBUG)
            {
                std::cout << "block_lines is null" << std::endl;
            }
            block_lines->push_back(std::move(std::make_unique<ASTPointer>(block_parser.parse_line(tokens))));

            // get input for block
            std::cout << ">>    ";
            std::getline(std::cin, input);

            tokens = lexer.tokenize(input);
        }

        return block_lines;
    }

    ASTPointer parse_assignment();
    ASTPointer parse_function_definition();
    std::vector<ASTPointer> parse_statement_list();

    std::vector<Token> tokens_;
    size_t current_token_index_;
    bool in_block_ = false;
    static int numBlocks_;
};

int Parser::numBlocks_ = 0;

#endif // PARSER_HPP