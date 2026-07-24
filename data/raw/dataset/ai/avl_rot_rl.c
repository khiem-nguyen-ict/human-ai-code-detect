#include "avl_rot_rl.h"
#include "avl_rot_left.h"
#include "avl_rot_right.h"

Node *avl_rotate_right_left(Node *node) {
    node->right = avl_rotate_right(node->right);
    return avl_rotate_left(node);
}
