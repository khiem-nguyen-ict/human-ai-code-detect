#include "avl_balance.h"
#include "avl_height.h"
int avl_get_balance(Node *node) {
    return node ? avl_get_height(node->left) - avl_get_height(node->right) : 0;
}
