#include "avl_rebalance.h"
#include "avl_balance.h"
#include "avl_rot_left.h"
#include "avl_rot_right.h"
#include "avl_rot_lr.h"
#include "avl_rot_rl.h"

Node *avl_rebalance_node(Node *node, int key) {
    int balance = avl_get_balance(node);

    if (balance > 1 && key < node->left->key)
        return avl_rotate_right(node);

    if (balance < -1 && key > node->right->key)
        return avl_rotate_left(node);

    if (balance > 1 && key > node->left->key)
        return avl_rotate_left_right(node);

    if (balance < -1 && key < node->right->key)
        return avl_rotate_right_left(node);

    return node;
}
