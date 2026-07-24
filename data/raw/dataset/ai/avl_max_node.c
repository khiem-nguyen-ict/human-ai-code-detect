#include "avl_max_node.h"

Node *avl_max_value_node(Node *node) {
    Node *current = node;
    while (current && current->right != NULL)
        current = current->right;
    return current;
}
