# Random Notes for Modern C++



## Lambda Expressions in C++
A lambda expression is as an anonymous, 'inline' function pointer that can carry its own local variables with it.

### 1. Basic Syntax

```cpp
auto my_lambda = [capture](parameters) -> return_type {
    // function body
}
```

### 2. Capture Clause
- `[ ]`: No variables are captured.
- `[=]`: All variables are captured by value.
- `[&]`: All variables are captured by reference.
- `[x, &y]`: `x` is captured by value, `y` by reference.

### 3. Example
We use a vector which is captured to distinguish `&`, `=` and `mutable`.
#### Capture by Reference (&)

```cpp
std::vector<int> nums = {1, 2, 3};

auto work_on_original = [&nums]() {
    nums.push_back(4); // Modifies the original vector in main()
};

work_on_original();
// nums.size() is now 4.
```

#### Capture by Value (=)

```cpp
std::vector<int> nums = {1, 2, 3};

auto work_on_copy = [nums]() {
    // nums.push_back(4); // ERROR! The copy is 'const' (read-only) by default.
    std::cout << "Copy size: " << nums.size() << std::endl;
};

work_on_copy();
// Original nums.size() is still 3.
```

#### Capture by Value with `mutable`

```cpp
std::vector<int> nums = {1, 2, 3};

auto experiment = [nums]() mutable {
    nums.push_back(99); // Works! Modifies the PRIVATE copy inside the lambda
    std::cout << "Inside lambda: " << nums.size() << std::endl; // Prints 4
};

experiment();
std::cout << "Outside: " << nums.size() << std::endl; // Still prints 3
```

#### Puzzle lambda expression as a member function
The following code snippet actually didn't capture anything. You may confuse why it can access `a` and `b` without capturing them. The reason is that the lambda is defined inside a member function, so it has access to the member variables of the class. So it just hide the `this` pointer.

```cpp
struct zip_iterator
{
    int *a;
    int *b;

    std::tuple<int, int> operator[](int i)
    {
        return {a[i], b[i]};
        // Real: return {this->a[i], this->b[i]};
    }
};
```

## Explain the std functions by C




### `std::move`
Transfer the ownership of resources from one object to another without making a copy.

```cpp
string n = "Player1";
string name = std::move(n); // Transfers ownership of n's resources to name
```

```c
char* n = malloc(100 * sizeof(char));
char* name = n; 
n=NULL; 
```

### `std::copy`
Copies elements from one range to another.

```cpp
std::vector<int> source = {1, 2, 3, 4, 5};
std::vector<int> destination(5);
std::copy(source.begin(), source.end(), destination.begin());
```

```c
int source[] = {1, 2, 3, 4, 5};
int destination[5];
for(int i = 0; i < 5; i++) {
    destination[i] = source[i];
}
```

### `std::swap`
Exchanges the values of two objects.

```cpp
int a = 5, b = 10;
std::swap(a, b); // Now a is 10 and b is 5
```

```c
int a = 5, b = 10, temp;
temp = a;
a = b;
b = temp; // Now a is 10 and b is 5
```

### `std::transform`
Applies a function to a range of elements and stores the result in another range.

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};
std::vector<int> result(5);
std::transform(vec.begin(), vec.end(), result.begin(), [](int x) { return x * 2; });
```

```c
int vec[] = {1, 2, 3, 4, 5};
int result[5];
for(int i = 0; i < 5; i++) {
    result[i] = vec[i] * 2;
}
```

### `std::reduce`
Header file: `<numeric>`
Combines elements in a range using a binary operation, producing a single result.
Operations are listed in the following:
- Sum (default)
- Product (using `std::multiplies<T>{}`)
- Minimum (using lambda expression`[](int a, int b){ return std::min(a, b); }`)

> `std::multiplies<int>{}` works but `std::max<int>{}` doesn't because one is a functor class, the other is a function template.
From `<functional> `you get:
`std::plus<T>{}, std::minus<T>{}, std::multiplies<T>{}, std::divides<T>{}
std::equal_to<T>{}, std::not_equal_to<T>{}, std::less<T>{}, std::greater<T>{}`
No `std::max<T>` functor exists because `std::max(a,b)` returns a reference to the larger argument, while functors need to return by value for reduction algorithms.

```cpp
std::vector<int> data = {1, 2, 3, 4, 5};
//sum
std::reduce(data.begin(), data.end(), 10); // Returns 25 (10 + sum of elements)
//product
std::reduce(data.begin(), data.end(), 1, std::multiplies<int>{}); // Returns 120 (product of elements)


```

### `std::unique_ptr`
A smart pointer that owns and manages another object through a pointer and disposes of that object when the `std::unique_ptr` goes out of scope.

```cpp
std::unique_ptr<int> ptr1(new int(10)); // Create a unique_ptr
std::unique_ptr<int> ptr2 = std::move(ptr1); // Transfer ownership to ptr2
```

```c
int* ptr1 = (int*)malloc(sizeof(int));
*ptr1 = 10; // Create a pointer
int* ptr2 = ptr1; // Transfer ownership to ptr2
ptr1 = NULL; // Avoid dangling pointer
```

### `std::shared_ptr`
A smart pointer that retains shared ownership of an object through a pointer. Several `std::shared_ptr` objects may own the same object. The object is destroyed when the last remaining `std::shared_ptr` owning the object is destroyed or reset.

```cpp
std::shared_ptr<int> ptr1 = std::make_shared<int>(20); // Create a shared_ptr
std::shared_ptr<int> ptr2 = ptr1; // Shared ownership
```

```c
int* ptr1 = (int*)malloc(sizeof(int));
*ptr1 = 20; // Create a pointer
int* ptr2 = ptr1; // Shared ownership
```

### `std::weak_ptr`
A smart pointer that holds a non-owning ("weak") reference to an object that is managed by `std::shared_ptr`. It must be converted to `std::shared_ptr` in order to access the referenced object.

```cpp
std::shared_ptr<int> sptr = std::make_shared<int>(30);
std::weak_ptr<int> wptr = sptr; // Create a weak_ptr from shared_ptr
```

```c
int* sptr = (int*)malloc(sizeof(int));
*sptr = 30; // Create a pointer
int* wptr = sptr; // Create a weak reference (no ownership)
```