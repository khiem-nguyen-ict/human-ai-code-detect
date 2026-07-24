#include "avl_tree_search.h"
#include "avl_search_node.h"

Node *avl_search(const AVLTree *tree, int key) {
    if (!tree) return NULL;
    return avl_search_node_internal(tree->root, key);
}
