#include "avl_insert_node.h"
#include "avl_node.h"
#include "avl_height.h"
#include "avl_max.h"
#include "avl_rebalance.h"

Node *avl_insert_node_internal(Node *node, int key, int *inserted) {
    if (!node) {
        *inserted = 1;
        return avl_create_node(key);
    }

    if (key < node->key) {
        node->left = avl_insert_node_internal(node->left, key, inserted);
    } else if (key > node->key) {
        node->right = avl_insert_node_internal(node->right, key, inserted);
    } else {
        *inserted = 0;
        return node;
    }

    node->height = 1 + avl_max(avl_get_height(node->left), avl_get_height(node->right));
    return avl_rebalance_node(node, key);
}
