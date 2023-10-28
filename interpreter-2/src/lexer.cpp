
#include "../includes/lexer.hpp"

Lexer::Lexer(std::string source) : source(source)
{
}

std::shared_ptr<std::vector<Token>> Lexer::tokenize()
{
    tokens = std::make_shared<std::vector<Token>>();

    // Iterate through the source line
    while (current < source.length())
    {
        start = current;
        scan_token();
    }

    // Add EOF token
    add_token(TokenType::END_OF_FILE);

    return tokens;
}

bool Lexer::increment_current()
{
    if (current < source.length())
    {
        current++;
        return true;
    }
    return false;
}

bool Lexer::is_alpha()
{
    char c = source[current];
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool Lexer::is_digit()
{
    char c = source[current];
    return c >= '0' && c <= '9';
}

void Lexer::scan_token()
{
    char c = source[current];
    if (!increment_current())
        return;

    if (is_alpha())
    {
        identifier();
    }
    else if (is_digit())
    {
        number();
    }
    else if (isspace(c))
    {
        return;
    }
    else
    {
        special_char();
    }
}

void Lexer::identifier()
{
    while (is_alpha() || is_digit())
    {
        increment_current();
    }

    std::string literal = source.substr(start, current - start);
    TokenType type = TokenType::IDENTIFIER;

    if (literal == "for")
        type = TokenType::FOR;
    else if (literal == "if")
        type = TokenType::IF;
    else if (literal == "else")
        type = TokenType::ELSE;
    else if (literal == "while")
        type = TokenType::WHILE;
    else if (literal == "return")
        type = TokenType::RETURN;
    else if (literal == "elseif")
        type = TokenType::ELSE_IF;
    else
        type = TokenType::IDENTIFIER;

    add_token(type, literal);
}

void Lexer::number()
{
    int dot_count = 0;
    Token token;
    token.type = TokenType::NUMBER;
    while (is_digit() || peek() == '.')
    {
        if (peek() == '.')
        {
            dot_count++;
            token.type = TokenType::FLOAT;
            if (dot_count > 1)
            {
                throw LexerError(current, "Invalid number.");
            }
        }
        increment_current();
    }

    std::string literal = source.substr(start, current - start);
    add_token(token.type, literal);
}

void Lexer::add_token(TokenType type)
{
    std::string literal = source.substr(start, current - start);
    Token token;
    token.type = type;
    token.lexeme = literal;
    tokens->push_back(token);
}

void Lexer::add_token(TokenType type, std::string literal)
{
    Token token;
    token.type = type;
    token.lexeme = literal;
    tokens->push_back(token);
}

bool Lexer::match(char expected)
{
    if (current >= source.length())
        return false;
    if (source[current] != expected)
        return false;
    return true;
}

bool Lexer::match_next(char expected)
{
    if (peek_next() == expected)
    {
        increment_current();
        return true;
    }
    return false;
}

char Lexer::peek()
{
    if (current >= source.length())
        return '\0';
    return source[current];
}

char Lexer::peek_next()
{
    if ((current + 1) >= source.length())
        return '\0';
    return source[current + 1];
}

void Lexer::special_char()
{

    char c = source[current];

    switch (c)
    {
    case '(':
        add_token(TokenType::LPAREN);
        break;
    case ')':
        add_token(TokenType::RPAREN);
        break;
    case '+':
        match_next('=') ? add_token(TokenType::ASSIG_ADD, "+=")
                        : add_token(TokenType::PLUS);
        break;
    case '-':
        match_next('=') ? add_token(TokenType::ASSIG_SUB, "-=")
                        : add_token(TokenType::MINUS);
        break;
    case '*':
        match_next('=') ? add_token(TokenType::ASSIG_MUL, "*=")
                        : add_token(TokenType::STAR);
        break;
    case '/':
        match_next('=') ? add_token(TokenType::ASSIG_DIV, "/=")
                        : add_token(TokenType::DIV);
        break;
    case ':':
        add_token(TokenType::COLON);
        break;

    case '=':
        match_next('=') ? add_token(TokenType::EQUAL_EQUAL, "==")
                        : add_token(TokenType::EQUAL);
        break;
    case '!':
        match_next('=') ? add_token(TokenType::NOT_EQUAL, "!=")
                        : add_token(TokenType::NOT);
        break;
    case '&':
        match_next('&') ? add_token(TokenType::AND, "&&")
                        : add_token(TokenType::BITWISE_AND);
        break;
    case '|':
        match_next('|') ? add_token(TokenType::OR, "||")
                        : add_token(TokenType::BITWISE_OR);
        break;
    case '<':
        match('=') ? add_token(TokenType::LESS_THAN_OR_EQUAL, "<=")
                   : add_token(TokenType::LESS_THAN);
        break;
    case '>':
        match('=') ? add_token(TokenType::GREATER_THAN_OR_EQUAL, ">=")
                   : add_token(TokenType::GREATER_THAN);
        break;
    case '.':
        match_next('*')   ? add_token(TokenType::ELEWISE_MUL, ".*")
        : match_next('/') ? add_token(TokenType::ELEWISE_DIV, "./")
        : match_next('^') ? add_token(TokenType::ELEWISE_POW, ".^")
        : match_next('%') ? add_token(TokenType::ELEWISE_MOD, ".%")
                          : add_token(TokenType::DOT);
        break;
    case '^':
        add_token(TokenType::POWER);
        break;
    case ',':
        add_token(TokenType::COMMA);
        break;
    case '%':
        match_next('=') ? add_token(TokenType::ASSIG_MOD, "%=")
                        : add_token(TokenType::MOD);
        break;
    default:
        throw LexerError(current, "Unexpected character.");
        break;
    }
}