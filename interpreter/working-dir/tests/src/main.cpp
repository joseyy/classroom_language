#include <iostream>
#include <string>
#include <iomanip>

int main()
{
    std::string input;
    std::cout << "Enter a word:" << std::endl;
    while (true)
    {
        // Print the initial message

        std::cout << input << std::flush;

        // Read input from the user
        std::cin >> input;

        // Check if the input is "end" and break the loop if needed
        if (input == "end")
        {
            break;
        }

        // Append space characters
        input = ' ' + input;
    }

    return 0;
}
