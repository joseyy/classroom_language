#ifndef INTERPRETER_HPP
#define INTERPRETER_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "./interpreter_error.hpp"
#include "./lexer.hpp"
#include "./parser.hpp"
// #include "./codeGenerator.hpp"

class Interpreter
{
public:
    Interpreter(std::ofstream &file) : file_(file) {}

    std::string execute(const std::string &input)
    {
        try
        {
            // Tokenize input
            Lexer lexer;
            std::vector<Token> tokens = lexer.tokenize(input);

            // Parse tokens into AST
            Parser parser;
            ASTPointer ast = parser.parse_line(tokens);
            /*
                        // Generate code from AST
                        CodeGenerator codeGenerator;
                        std::string code = codeGenerator.generate(*ast);

                        // check file is open
                        if (!file_.is_open())
                        {
                            throw InterpreterError("File not open");
                        }

                        file_ << code;
            */
            if (input == "error")
            {
                throw InterpreterError("Interpreter error");
            }
        }
        catch (InterpreterError *e)
        {
            throw e;
        }
        catch (std::exception &e)
        {
            throw e;
        }

        return input;
    }

private:
    std::ofstream &file_;
};

#endif // INTERPRETER_HPP