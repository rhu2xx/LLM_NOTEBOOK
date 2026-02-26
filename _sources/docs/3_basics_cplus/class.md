# Modern C++

## 1. The Basic Structure
```cpp
class Player {
private:
    std::string name; // Member variable
    int health{100};  // Default member initializer (Modern C++)

public:
    // Constructor
    Player(std::string n) : name{std::move(n)} {} 

    // Method (const means it doesn't modify the object)
    void print_status() const {
        std::cout << name << " has " << health << " HP." << std::endl;
    }
};
```

## 2. The rule of Three/Five/Zero
When you define one of the following, you should usually define all three/five:
- Destructor `~ClassName()`
- Copy Constructor `ClassName(const ClassName &other)`
- Copy Assignment Operator `ClassName& operator=(const ClassName &other)`
- Move Constructor (C++11) `ClassName(ClassName &&other) noexcept`
- Move Assignment Operator (C++11) `ClassName& operator=(ClassName &&other)`

`Constructor` use the member initializer list (`:name{n}`)
## 3. Access Modifiers
- `public`: Accessible from anywhere.
- `private`: Accessible only within the class.
- `protected`: Accessible within the class and its derived classes.


## 4. Modern Keywords to Remember
- `auto`: Let the compiler deduce types.

- `nullptr`: Always use this instead of NULL or 0.

- `override`: Explicitly tell the compiler you are overriding a virtual function.
- `final`: Prevent a class from being inherited or a method from being overridden.

- `default` / `delete`: Tell the compiler to create a default version of a function or explicitly forbid it (e.g., `Player(const Player&) = delete;` makes a class non-copyable).


## Example Usage
```cpp
#include <iostream>
#include <string>
#include <vector>

class BankAccount{
private:
    std::string owner;
    double balance{0.0};
    std::vector<std::string> log;

public:
    // `explicit`: Only use it if the programmer explicitly asks for it.`
    explicit BankAccount(std::string name, double initial_deposit): owner{std::move(name)}, balance{initial_deposit} {
        log.push_back("Account created for " + owner);
    }

}
```


