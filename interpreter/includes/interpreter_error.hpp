#ifndef INTERPRETER_ERROR_HPP
#define INTERPRETER_ERROR_HPP


#include <string>
#include <iostream>

class InterpreterError
{
public:
    InterpreterError(const std::string &message) : message_(message)
    {
    }

    const char *what() const
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

#endif // INTERPRETER_ERROR_HPP