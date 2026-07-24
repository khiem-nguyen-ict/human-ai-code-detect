#include "avl_postorder.h"
#include <stdio.h>

void avl_postorder_traversal(Node *root) {
    if (root) {
        avl_postorder_traversal(root->left);
        avl_postorder_traversal(root->right);
        printf("%d ", root->key);
    }
}
