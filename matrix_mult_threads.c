#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// Matriz contígua
// Toda a matriz vive em um único bloco de memória: data[i*n + j]

typedef struct {
    double* data; // bloco contíguo de n*n doubles
    int     n;
} Matrix;

// Macro de acesso: M(mat, i, j) equivale a mat.data[i*n + j]
#define M(mat, i, j) ((mat).data[(i) * (mat).n + (j)])

Matrix alloc_matrix(int n) {
    Matrix m;
    m.n    = n;
    m.data = (double*)calloc((size_t)n * n, sizeof(double));
    if (!m.data) { fprintf(stderr, "Erro: sem memória para matriz %dx%d\n", n, n); exit(1); }
    return m;
}

void free_matrix(Matrix* m) {
    free(m->data);
    m->data = NULL;
}

void fill_random(Matrix* m) {
    size_t total = (size_t)m->n * m->n;
    for (size_t i = 0; i < total; i++)
        m->data[i] = rand() % 10;
}

void print_matrix(Matrix* m) {
    for (int i = 0; i < m->n; i++) {
        for (int j = 0; j < m->n; j++)
            printf("%6.1f ", M(*m, i, j));
        printf("\n");
    }
}

// Estrutura passada para cada thread 
typedef struct {
    Matrix* A;
    Matrix* Bt;       // B já transposta
    Matrix* C;
    int     row_start;
    int     row_end;
} ThreadArgs;

// Versão sequencial (referência) 
void matrix_multiply_seq(Matrix* A, Matrix* B, Matrix* C) {
    int n = A->n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += M(*A, i, k) * M(*B, k, j);
            M(*C, i, j) = sum;
        }
}

// Transpõe B → Bt (acesso de coluna vira acesso de linha) 
Matrix transpose(Matrix* B) {
    int n = B->n;
    Matrix Bt = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M(Bt, i, j) = M(*B, j, i);
    return Bt;
}

// Função executada por cada thread
// Cada thread calcula as linhas [row_start, row_end) de C.
// Usa Bt (B transposta) para que ambos A e Bt sejam acessados por linha,
// maximizando hits de cache. Sem mutex pois linhas são independentes.
void* thread_worker(void* arg) {
    ThreadArgs* a  = (ThreadArgs*)arg;
    Matrix*     A  = a->A;
    Matrix*     Bt = a->Bt;
    Matrix*     C  = a->C;
    int         n  = A->n;

    for (int i = a->row_start; i < a->row_end; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += M(*A, i, k) * M(*Bt, j, k); // Bt[j][k] = B[k][j]
            M(*C, i, j) = sum;
        }
    }
    return NULL;
}

// Multiplicação paralela 
void matrix_multiply_parallel(Matrix* A, Matrix* B, Matrix* C, int num_threads) {
    int n = A->n;

    // Transpõe B uma única vez, fora das threads
    Matrix Bt = transpose(B);

    pthread_t*  threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* args    = malloc(num_threads * sizeof(ThreadArgs));

    int rows_per_thread = n / num_threads;
    int remainder       = n % num_threads;
    int row = 0;

    for (int t = 0; t < num_threads; t++) {
        int extra         = (t < remainder) ? 1 : 0;
        args[t].A         = A;
        args[t].Bt        = &Bt;
        args[t].C         = C;
        args[t].row_start = row;
        args[t].row_end   = row + rows_per_thread + extra;
        row = args[t].row_end;
        pthread_create(&threads[t], NULL, thread_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    free_matrix(&Bt);
    free(threads);
    free(args);
}

int main(int argc, char* argv[]) {
    int n           = 512;
    int num_threads = 4;

    if (argc > 1) n           = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);

    if (n <= 0 || num_threads <= 0) {
        fprintf(stderr, "Uso: %s <tamanho> <threads>\n", argv[0]);
        return 1;
    }

    printf("=== Multiplicação de Matrizes com Threads ===\n");
    printf("Tamanho  : %d x %d\n", n, n);
    printf("Threads  : %d\n", num_threads);
    printf("Memória  : ~%.1f MB por matriz\n\n",
           (double)n * n * sizeof(double) / (1024 * 1024));

    srand(42);

    Matrix A     = alloc_matrix(n);
    Matrix B     = alloc_matrix(n);
    Matrix C_seq = alloc_matrix(n);
    Matrix C_par = alloc_matrix(n);

    fill_random(&A);
    fill_random(&B);

    struct timespec t0, t1;

    // Sequencial
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matrix_multiply_seq(&A, &B, &C_seq);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tempo_seq = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Paralelo
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matrix_multiply_parallel(&A, &B, &C_par, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tempo_par = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Resultados 
    printf("Tempo sequencial : %.4f s\n", tempo_seq);
    printf("Tempo paralelo   : %.4f s\n", tempo_par);
    printf("Speedup          : %.2fx\n", tempo_seq / tempo_par);
    printf("Eficiência       : %.2f%%\n\n", (tempo_seq / tempo_par) / num_threads * 100.0);

    // Verificação de corretude 
    int ok = 1;
    for (int i = 0; i < n && ok; i++)
        for (int j = 0; j < n && ok; j++)
            if (C_seq.data[i * n + j] != C_par.data[i * n + j]) ok = 0;
    printf("Verificação      : %s\n", ok ? "OK — resultados idênticos ✓"
                                         : "ERRO — resultados divergem ✗");

    if (n <= 8) {
        printf("\nMatriz A:\n");       print_matrix(&A);
        printf("\nMatriz B:\n");       print_matrix(&B);
        printf("\nC (sequencial):\n"); print_matrix(&C_seq);
        printf("\nC (paralela):\n");   print_matrix(&C_par);
    }

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C_seq);
    free_matrix(&C_par);

    return 0;
}