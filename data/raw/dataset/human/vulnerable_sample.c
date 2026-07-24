#include <string.h>
#include <stdlib.h>

void vulnerable_copy(char *input) {
    char buffer[64];
    strcpy(buffer, input);
}

int main(int argc, char **argv) {
    vulnerable_copy(argv[1]);
    return 0;
}
