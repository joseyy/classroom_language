#ifndef LEXER_HPP
#define LEXER_HPP

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "./token.hpp"
#include "./lexer_error.hpp"

class Lexer
{
public:
    Lexer(std::string source) : source(source) {}
    std::shared_ptr<std::vector<Token>> tokenize();

private:
    std::string source;
    std::shared_ptr<std::vector<Token>> tokens;
    int start = 0;
    int current = 0;

    bool increment_current();
    void add_token(TokenType type);
    void add_token(TokenType type, std::string literal);
    bool match(char expected);
    bool match_next(char expected);
    char peek();
    char peek_next();
    bool is_digit();
    void number();
    bool is_alpha();
    void identifier();
    void scan_token();
    void special_char();
};

#endif // LEXER_HPP