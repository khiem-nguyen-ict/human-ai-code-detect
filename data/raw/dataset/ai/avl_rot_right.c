#include "avl_rot_right.h"
#include "avl_height.h"
#include "avl_max.h"

Node *avl_rotate_right(Node *y) {
    Node *x = y->left;
    Node *T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = avl_max(avl_get_height(y->left), avl_get_height(y->right)) + 1;
    x->height = avl_max(avl_get_height(x->left), avl_get_height(x->right)) + 1;

    return x;
}
