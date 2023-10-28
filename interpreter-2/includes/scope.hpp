#ifndef SCOPE_HPP
#define SCOPE_HPP

#include <string>
#include <stack>
#include <memory>
#include <iostream>
#include <unordered_map>

template <typename T>
struct VariableInScope
{
    std::string name;
    std::string type;
    T value;
};

template <typename T>   
struct Scope
{
    std::unordered_map<std::string, VariableInScope<T>> variables;
    std::shared_ptr<Scope> parent;
};

template <typename T>
class ScopeManager
{
public:
    ScopeManager();
    void push_scope();
    void pop_scope();
    void add_variable(std::string name, std::string type, std::string value);
    void get_variable(std::string name);

private:
    std::shared_ptr<std::stack<std::shared_ptr<Scope>>> scope_stack;
};

#endif // SCOPE_HPP