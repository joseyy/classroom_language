#ifndef TOKEN_HPP
#define TOKEN_HPP

#include <string>
#include <vector>

enum class TokenType
{
    // Single Keywords
    FOR,    // for keyword
    IF,     // if keyword
    ELSE,   // else keyword
    WHILE,  // while keyword
    RETURN, // return keyword

    // Multi Keywords
    ELSE_IF,

    // single character tokens
    PLUS,         // +
    STAR,         // *
    MINUS,        // -
    EQUAL,        // =
    GREATER_THAN, // >
    LESS_THAN,    // <
    NOT,          // !
    BITWISE_AND,  // &
    BITWISE_OR,   // |
    COLON,        // :
    LPAREN,       // (
    RPAREN,       // )
    DOT,          // .
    POWER,        // ^
    COMMA,        // ,
    MOD,          // %
    DIV,          // /

    // Compound character tokens
    LESS_THAN_OR_EQUAL,    // <=
    GREATER_THAN_OR_EQUAL, // >=
    EQUAL_EQUAL,           // ==
    NOT_EQUAL,             // !=
    OR,                    // ||
    AND,                   // &&
    ELEWISE_MUL,           // .*
    ELEWISE_DIV,           // ./
    ELEWISE_POW,           // .^
    ELEWISE_MOD,           // .%
    ASSIG_MUL,             // *=
    ASSIG_DIV,             // /=
    ASSIG_ADD,             // +=
    ASSIG_SUB,             // -=
    ASSIG_POW,             // ^=
    ASSIG_MOD,             // %=

    // Literals
    IDENTIFIER,
    NUMBER,
    FLOAT,

    // Other
    END_OF_FILE,   
};

struct Token
{
    TokenType type;
    std::string lexeme;
};

#endif // TOKEN_HPP