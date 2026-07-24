#include "avl_print.h"
#include "avl_inorder.h"
#include "avl_preorder.h"
#include "avl_postorder.h"
#include <stdio.h>

void avl_print_inorder(const AVLTree *tree) {
    if (!tree) return;
    avl_inorder_traversal(tree->root);
    printf("\n");
}

void avl_print_preorder(const AVLTree *tree) {
    if (!tree) return;
    avl_preorder_traversal(tree->root);
    printf("\n");
}

void avl_print_postorder(const AVLTree *tree) {
    if (!tree) return;
    avl_postorder_traversal(tree->root);
    printf("\n");
}
