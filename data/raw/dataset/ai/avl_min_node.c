#include "avl_min_node.h"

Node *avl_min_value_node(Node *node) {
    Node *current = node;
    while (current && current->left != NULL)
        current = current->left;
    return current;
}
