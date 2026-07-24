#include "avl_height.h"
int avl_get_height(Node *node) {
    return node ? node->height : 0;
}
