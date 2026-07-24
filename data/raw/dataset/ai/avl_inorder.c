#include "avl_inorder.h"
#include <stdio.h>

void avl_inorder_traversal(Node *root) {
    if (root) {
        avl_inorder_traversal(root->left);
        printf("%d ", root->key);
        avl_inorder_traversal(root->right);
    }
}
