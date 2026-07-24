#include "avl_tree_delete.h"
#include "avl_delete_node.h"

void avl_delete(AVLTree *tree, int key) {
    if (!tree) return;
    int deleted = 0;
    tree->root = avl_delete_node_internal(tree->root, key, &deleted);
    if (deleted) tree->size--;
}
