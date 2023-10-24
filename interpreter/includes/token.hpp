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
    END_OF_FILE,

};

struct Token
{
    TokenType type;
    std::string value;
};

#endif // TOKEN_HPP