#include "avl_search_node.h"

Node *avl_search_node_internal(Node *root, int key) {
    Node *curr = root;
    while (curr) {
        if (key == curr->key) return curr;
        curr = (key < curr->key) ? curr->left : curr->right;
    }
    return NULL;
}
