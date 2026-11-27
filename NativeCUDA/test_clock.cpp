#include <chrono>
#include <type_traits>
#include <stdio.h>
int main() {
    printf("is_same: %d\n", std::is_same<std::chrono::high_resolution_clock, std::chrono::steady_clock>::value);
    printf("high_res is_steady: %d\n", std::chrono::high_resolution_clock::is_steady);
    printf("steady is_steady: %d\n", std::chrono::steady_clock::is_steady);
    return 0;
}
