#include <stdio.h>
#include "avl_tree_init.h"
#include "avl_tree_destroy.h"
#include "avl_tree_insert.h"
#include "avl_tree_delete.h"
#include "avl_tree_search.h"
#include "avl_print.h"

int main(void) {
    AVLTree *tree = avl_create();

    int keys[] = {10, 20, 30, 40, 50, 25};
    size_t n = sizeof(keys) / sizeof(keys[0]);

    printf("Inserting keys: ");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", keys[i]);
        avl_insert(tree, keys[i]);
    }
    printf("\n");

    printf("Inorder traversal: ");
    avl_print_inorder(tree);

    printf("Preorder traversal: ");
    avl_print_preorder(tree);

    printf("Deleting key 30...\n");
    avl_delete(tree, 30);

    printf("Inorder traversal after deletion: ");
    avl_print_inorder(tree);

    printf("Searching for 25: %s\n", avl_search(tree, 25) ? "Found" : "Not Found");
    printf("Searching for 30: %s\n", avl_search(tree, 30) ? "Found" : "Not Found");

    avl_destroy(tree);
    return 0;
}
