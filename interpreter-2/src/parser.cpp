#include "../includes/parser.hpp"

Parser::Parser(std::shared_ptr<std::vector<Token>> tokens) : tokens(tokens)
{
}

bool Parser::is_at_end()
{
    if (current >= tokens->size())
    {
        return true;
    }
    return false;
}

Token Parser::peek()
{
    return tokens->at(current);
}

Token Parser::peek_next()
{
    if (is_at_end())
    {
        return tokens->at(current);
    }
    return tokens->at(current + 1);
}

Token Parser::advance()
{
    if (!is_at_end())
    {
        current++;
    }
    // This will advance the current token and return the previous one
    return previous();
}

Token Parser::previous()
{
    return tokens->at(current - 1);
}

bool Parser::check(TokenType type)
{
    if (is_at_end())
    {
        return false;
    }
    return peek().type == type;
}

Token Parser::consume(TokenType type, std::string message)
{
    if (check(type))
    {
        return advance();
    }
}