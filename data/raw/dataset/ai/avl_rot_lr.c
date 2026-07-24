#include "avl_rot_lr.h"
#include "avl_rot_left.h"
#include "avl_rot_right.h"

Node *avl_rotate_left_right(Node *node) {
    node->left = avl_rotate_left(node->left);
    return avl_rotate_right(node);
}
