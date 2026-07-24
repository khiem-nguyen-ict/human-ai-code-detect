#include "avl_tree_destroy.h"
#include "avl_free_node.h"
#include <stdlib.h>

void avl_destroy(AVLTree *tree) {
    if (tree) {
        avl_free_nodes(tree->root);
        free(tree);
    }
}
