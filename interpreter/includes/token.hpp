#ifndef TOKEN_HPP
#define TOKEN_HPP

#include <string>

enum class TokenType
{
    NUMBER,
    FLOAT,
    PLUS,
    MINUS,
    MUL,
    DIV,
    LPAREN,
    RPAREN,
    IDENTIFIER,
    DOT,
    END_OF_LINE,
    ASSIGNMENT,
    FUNCTION,
    PRINT,
    IF,
    ELSE,
    ELSEIF,
    FOR,
    END,
    COMMA,
    BRAKET_OPEN,
    BRAKET_CLOSE,
    COLON,
    GREATER_THAN,
    LESS_THAN,
    GREATER_THAN_EQUAL,
    LESS_THAN_EQUAL,
    EQUAL,
    NOT_EQUAL,
    END_OF_FILE,

};

struct Token
{
    TokenType type;
    std::string value;
};

#endif // TOKEN_HPP