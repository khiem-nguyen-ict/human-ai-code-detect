#include <string.h>
#include <stdio.h>
void unsafe_copy(char *src) {
    char dst[32];
    strcpy(dst, src);
}
int main() { return 0; }
