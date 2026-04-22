#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Aloca uma matriz n x n com zeros
double** alloc_matrix(int n) {
    double** m = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        m[i] = (double*)calloc(n, sizeof(double));
    }
    return m;
}

// Libera a memória de uma matriz n x n
void free_matrix(double** m, int n) {
    for (int i = 0; i < n; i++) free(m[i]);
    free(m);
}

// Preenche a matriz com valores aleatórios entre 0 e 9
void fill_random(double** m, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = rand() % 10;
}

// Imprime a matriz 
void print_matrix(double** m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%6.1f ", m[i][j]);
        printf("\n");
    }
}

/*
 * Multiplicação de matrizes sequencial — O(n³)
 *
 * Para cada elemento C[i][j], percorre todos os k:
 *   C[i][j] = sum(A[i][k] * B[k][j]) para k = 0..n-1
 *
 * Três loops aninhados → complexidade O(n³)
 */
void matrix_multiply(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {           // linha de A / linha de C
        for (int j = 0; j < n; j++) {       // coluna de B / coluna de C
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {   // percorre a linha de A e a coluna de B
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int n = 512; // tamanho padrão
    if (argc > 1) n = atoi(argv[1]);

    printf("=== Multiplicação de Matrizes Sequencial O(n³) ===\n");
    printf("Tamanho: %d x %d\n\n", n, n);

    srand(42); // semente fixa para reprodutibilidade

    double** A = alloc_matrix(n);
    double** B = alloc_matrix(n);
    double** C = alloc_matrix(n);

    fill_random(A, n);
    fill_random(B, n);

    // --- Medição de tempo ---
    struct timespec inicio, fim;
    clock_gettime(CLOCK_MONOTONIC, &inicio);

    matrix_multiply(A, B, C, n);

    clock_gettime(CLOCK_MONOTONIC, &fim);

    double tempo = (fim.tv_sec - inicio.tv_sec)
                 + (fim.tv_nsec - inicio.tv_nsec) / 1e9;

    printf("Tempo de execução: %.4f segundos\n", tempo);
    printf("Operações realizadas: ~%.2e multiplicações\n", (double)n * n * n);

    // Exibe um pequeno trecho do resultado para verificação
    if (n <= 8) {
        printf("\nMatriz A:\n"); print_matrix(A, n);
        printf("\nMatriz B:\n"); print_matrix(B, n);
        printf("\nMatriz C = A x B:\n"); print_matrix(C, n);
    } else {
        printf("\nC[0][0] = %.1f  (verificação rápida)\n", C[0][0]);
    }

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}
