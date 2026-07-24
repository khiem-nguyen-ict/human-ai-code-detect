#include "avl_tree_init.h"
#include <stdlib.h>

AVLTree *avl_create(void) {
    AVLTree *tree = (AVLTree *)malloc(sizeof(AVLTree));
    if (tree) {
        tree->root = NULL;
        tree->size = 0;
    }
    return tree;
}
