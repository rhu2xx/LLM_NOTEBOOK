#include <cstdio>
#include <array>
#include <tuple>

struct counting_iterator{
    int operator[](int i){
        return i;
    }
};

struct transform_iterator{
    int *a;
    int operator[](int i){
        return a[i]*2;
    }

};

struct zip_iterator
{
    int *a;
    int *b;

    std::tuple<int, int> operator[](int i)
    {
        return {this->a[i], this->b[i]};
    }
};

int main(){
    counting_iterator it_count;
    std::printf("Output the counting_iterator values:\n");
    std::printf("Value at index 5 at it_count[5]: %d\n", it_count[5]);
    std::printf("Value at index 10 at it_count[10]: %d\n", it_count[10]);

    std::array<int, 3> x{0,1,2};
    transform_iterator it_transform{x.data()};
    std::printf("Output the counting_iterator values:\n");
    std::printf("Array values: %d, %d, %d\n", x[0], x[1], x[2]);;

    std::printf("Value at index 0 at it_transform[0]: %d\n", it_transform[0]);
    std::printf("Value at index 1 at it_transform[1]: %d\n", it_transform[1]);

    std::array<int, 3> a{ 0, 1, 2 };
    std::array<int, 3> b{ 5, 4, 2 };

    zip_iterator it{a.data(), b.data()};

    std::printf("it[0]: (%d, %d)\n", std::get<0>(it[0]), std::get<1>(it[0])); // prints (0, 5)
    std::printf("it[0]: (%d, %d)\n", std::get<0>(it[1]), std::get<1>(it[1])); // prints (1, 4)

}
