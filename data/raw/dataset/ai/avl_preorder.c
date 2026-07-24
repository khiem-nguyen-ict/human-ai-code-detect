#include "avl_preorder.h"
#include <stdio.h>

void avl_preorder_traversal(Node *root) {
    if (root) {
        printf("%d ", root->key);
        avl_preorder_traversal(root->left);
        avl_preorder_traversal(root->right);
    }
}
