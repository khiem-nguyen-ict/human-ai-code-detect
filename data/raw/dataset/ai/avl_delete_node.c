#include "avl_delete_node.h"
#include "avl_min_node.h"
#include "avl_height.h"
#include "avl_max.h"
#include "avl_balance.h"
#include "avl_rot_left.h"
#include "avl_rot_right.h"
#include <stdlib.h>

Node *avl_delete_node_internal(Node *root, int key, int *deleted) {
    if (!root) return root;

    if (key < root->key) {
        root->left = avl_delete_node_internal(root->left, key, deleted);
    } else if (key > root->key) {
        root->right = avl_delete_node_internal(root->right, key, deleted);
    } else {
        *deleted = 1;
        if (!root->left || !root->right) {
            Node *temp = root->left ? root->left : root->right;
            if (!temp) {
                temp = root;
                root = NULL;
            } else {
                *root = *temp;
            }
            free(temp);
        } else {
            Node *temp = avl_min_value_node(root->right);
            root->key = temp->key;
            root->right = avl_delete_node_internal(root->right, temp->key, deleted);
        }
    }

    if (!root) return root;

    root->height = 1 + avl_max(avl_get_height(root->left), avl_get_height(root->right));
    int balance = avl_get_balance(root);

    if (balance > 1 && avl_get_balance(root->left) >= 0)
        return avl_rotate_right(root);

    if (balance > 1 && avl_get_balance(root->left) < 0) {
        root->left = avl_rotate_left(root->left);
        return avl_rotate_right(root);
    }

    if (balance < -1 && avl_get_balance(root->right) <= 0)
        return avl_rotate_left(root);

    if (balance < -1 && avl_get_balance(root->right) > 0) {
        root->right = avl_rotate_right(root->right);
        return avl_rotate_left(root);
    }

    return root;
}
