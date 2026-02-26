
#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    float k = 0.5;
    float ambient_temp = 20;
    std::vector<float> temp{ 42, 24, 50 };
    

    auto op = [=](float temp){
        float diff = ambient_temp - temp;
        return temp + k * diff;
    };

    std::printf("step  temp[0]  temp[1]  temp[2]\n");
    for (int step = 0; step < 3; step++) {
        

        std::transform(temp.begin(), temp.end(),
                        temp.begin(), op);

        std::printf("%d     %.2f    %.2f    %.2f\n", step, temp[0], temp[1], temp[2]);
    }
}
