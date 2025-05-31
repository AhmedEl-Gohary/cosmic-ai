#include <stdio.h>

int sum(int x, int y) {
    int z = 0;
    z = x + y;
    return z;
}
void checkEvenOrOdd(int x) {
    if (x % 2 == 0)
        printf("%d is Even\n", x);
    else
        printf("%d is odd\n", x);
}
int square(int x) {
    return x * x;
}
int maximum(int x, int y) {
    int max = 0;

    if (x < y)
        max = y;
    else
        max = x;

    return max;
}

int main() {
    int x = 0, y = 0;
    printf("Give two integers numbers\n");
    scanf("%d%d", &x, &y);
    int z = 0, result;
    z = x + 2;
    result = z;
    int tmp = square(z);
    int su = sum(x, y);
    int max = maximum(x, y);
    result += 2;
    checkEvenOrOdd(result);
    printf("The sum of %d and %d is : %d\n", x, y, su);
    printf("The square of %d is : %d\n", z, tmp);
    printf("The maximum of %d and %d is : %d\n", x, y, max);

    return 0;
}



