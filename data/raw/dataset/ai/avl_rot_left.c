#include "avl_rot_left.h"
#include "avl_height.h"
#include "avl_max.h"

Node *avl_rotate_left(Node *x) {
    Node *y = x->right;
    Node *T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = avl_max(avl_get_height(x->left), avl_get_height(x->right)) + 1;
    y->height = avl_max(avl_get_height(y->left), avl_get_height(y->right)) + 1;

    return y;
}
