#ifndef LEXER_ERROR_HPP
#define LEXER_ERROR_HPP

#include <string>
#include <exception>

class LexerError : public std::exception
{
public:
    LexerError(char c, std::string message)
    {
        this->message = message + " '" + c + "'";
    }

    LexerError(std::string message)
    {
        this->message = message;
    }

    const char *what() const throw()
    {
        std::string error_message = "Lexer Error: " + message;
        return error_message.c_str();
    }

private:
    std::string message;
};

#endif // LEXER_ERROR_HPP