#ifndef LEXER_HPP
#define LEXER_HPP

#include <vector>
#include <string>
#include <sstream>

#include "interpreter_error.hpp"
#include "token.hpp"

class Lexer
{
private:
    std::string input_;
    std::vector<Token> tokens;

public:
    Lexer(const std::string &input) : input_(input)
    {
    }
    std::vector<Token> tokenize()
    {
        std::istringstream iss(input_);
        char c;
        while (iss >> c)
        {
            if (isdigit(c))
            {
                std::string number;
                number += c;
                while (isdigit(iss.peek()))
                {
                    iss >> c;
                    number += c;
                }
                tokens.push_back({TokenType::NUMBER, number});
            }
            else if (isalpha(c) || c == '_')
            {
                std::string identifier;
                identifier += c;
                while (isalnum(iss.peek()) || iss.peek() == '_')
                {
                    iss >> c;
                    identifier += c;
                }
                tokens.push_back({TokenType::IDENTIFIER, identifier});
            }
            else if (isspace(c))
            {
                continue;
            }
            else if (c == '+')
            {
                tokens.push_back({TokenType::PLUS, "+"});
            }
            else if (c == '-')
            {
                tokens.push_back({TokenType::MINUS, "-"});
            }
            else if (c == '*')
            {
                tokens.push_back({TokenType::MUL, "*"});
            }
            else if (c == '/')
            {
                tokens.push_back({TokenType::DIV, "/"});
            }
            else if (c == '(')
            {
                tokens.push_back({TokenType::LPAREN, "("});
            }
            else if (c == ')')
            {
                tokens.push_back({TokenType::RPAREN, ")"});
            }
            else if(c == '.')
            {
                tokens.push_back({TokenType::DOT, "."});
            }
            else if(c == ';')
            {
                tokens.push_back({TokenType::END_OF_LINE, ";"});
            }
            else
            {
                throw InterpreterError("Invalid character");
            }
        }

        return tokens;
    }
};

#endif // LEXER_HPP