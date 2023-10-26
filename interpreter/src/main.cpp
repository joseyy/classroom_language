#include <string>
#include <iostream>
#include <fstream>

#include "../includes/interpreter.hpp"
#include "../includes/interpreter_error.hpp"
#include "../includes/lexer.hpp"
#include "../includes/parser.hpp"

std::string pathToFile = "output.cpp";

int main()
{

    // Open file
    std::ofstream file;
    file.open(pathToFile);

    // Create interpreter
    Interpreter interpreter(file);

    while (true)
    {
        // Read input from user
        std::cout << ">> ";
        std::string input;
        std::getline(std::cin, input);

        // Check for exit command
        if (input == "exit")
        {
            break;
        }

        // Parse input and execute program
        try
        {
            std::string output = interpreter.execute(input);
            std::cout << output << std::endl;
        }
        catch (InterpreterError &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}