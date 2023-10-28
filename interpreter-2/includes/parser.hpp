#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "./lexer.hpp"
#include "./interpreter_error.hpp"
#include "./ast.hpp"


class Parser
{
public:
    Parser(std::shared_ptr<std::vector<Token>> tokens) : tokens(tokens) {}
    ASTPointer parse();

private:
    std::shared_ptr<std::vector<Token>> tokens;
    int current = 0;

    bool increment_current();
    bool match(TokenType type);
    bool match_next(TokenType type);
    Token peek();
    Token peek_next();
    Token previous();
    bool is_at_end();
    Token advance();
    bool check(TokenType type);
    Token consume(TokenType type, std::string message);
    void synchronize();
    ASTPointer expression();
};

#endif // PARSER_HPP