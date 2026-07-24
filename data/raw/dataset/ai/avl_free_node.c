#include "avl_free_node.h"
#include <stdlib.h>

void avl_free_nodes(Node *node) {
    if (node) {
        avl_free_nodes(node->left);
        avl_free_nodes(node->right);
        free(node);
    }
}
