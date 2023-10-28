#include "../includes/scope.hpp"

ScopeManager::ScopeManager()
{
    scope_stack = std::make_shared<std::stack<std::shared_ptr<Scope>>>();
    scope_stack->push(std::make_shared<Scope>());
}

void ScopeManager::push_scope()
{
    scope_stack->push(std::make_shared<Scope>());
}

void ScopeManager::pop_scope()
{
    scope_stack->pop();
}

void ScopeManager::add_variable(std::string name, 
                                std::string type, 
                                std::string value)
{
    VariableInScope var;
    var.name = name;
    var.type = type;
    var.value = value;
    scope_stack->top()->variables[name] = var;
}