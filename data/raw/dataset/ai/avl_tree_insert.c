#include "avl_tree_insert.h"
#include "avl_insert_node.h"

void avl_insert(AVLTree *tree, int key) {
    if (!tree) return;
    int inserted = 0;
    tree->root = avl_insert_node_internal(tree->root, key, &inserted);
    if (inserted) tree->size++;
}
