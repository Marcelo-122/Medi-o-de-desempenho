#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

// Matriz contígua 
typedef struct {
    double* data;
    int     n;
} Matrix;

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

// ─── Estrutura passada para cada thread ───────────────────────────────────────
typedef struct {
    Matrix* A;
    Matrix* Bt;
    Matrix* C;
    int     row_start;
    int     row_end;
} ThreadArgs;

// ─── Versão sequencial ────────────────────────────────────────────────────────
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

// ─── Transposta ───────────────────────────────────────────────────────────────
Matrix transpose(Matrix* B) {
    int n = B->n;
    Matrix Bt = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M(Bt, i, j) = M(*B, j, i);
    return Bt;
}

// ─── Worker das threads ───────────────────────────────────────────────────────
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
                sum += M(*A, i, k) * M(*Bt, j, k);
            M(*C, i, j) = sum;
        }
    }
    return NULL;
}

// ─── Versão paralela ──────────────────────────────────────────────────────────
void matrix_multiply_parallel(Matrix* A, Matrix* B, Matrix* C, int num_threads) {
    int n = A->n;

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

// ─── Uso ──────────────────────────────────────────────────────────────────────
void print_usage(const char* prog) {
    printf("Uso:\n");
    printf("  %s <tamanho> seq              — roda versão sequencial\n", prog);
    printf("  %s <tamanho> par <threads>    — roda versão paralela\n\n", prog);
    printf("Exemplos:\n");
    printf("  %s 1000 seq\n", prog);
    printf("  %s 1000 par 4\n", prog);
    printf("  %s 1000 par 12\n", prog);
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    int n = atoi(argv[1]);
    if (n <= 0) { fprintf(stderr, "Erro: tamanho inválido '%s'\n", argv[1]); return 1; }

    int modo_seq = (strcmp(argv[2], "seq") == 0);
    int modo_par = (strcmp(argv[2], "par") == 0);

    if (!modo_seq && !modo_par) {
        fprintf(stderr, "Erro: modo deve ser 'seq' ou 'par'\n");
        print_usage(argv[0]);
        return 1;
    }

    int num_threads = 0;
    if (modo_par) {
        if (argc < 4) { fprintf(stderr, "Erro: modo 'par' requer o número de threads\n"); return 1; }
        num_threads = atoi(argv[3]);
        if (num_threads <= 0) { fprintf(stderr, "Erro: número de threads inválido\n"); return 1; }
    }

    printf("=== Multiplicação de Matrizes ===\n");
    printf("Tamanho : %d x %d\n", n, n);
    printf("Modo    : %s", modo_seq ? "sequencial" : "paralelo");
    if (modo_par) printf(" (%d threads)", num_threads);
    printf("\nMemória : ~%.1f MB por matriz\n\n",
           (double)n * n * sizeof(double) / (1024 * 1024));

    srand(42);

    Matrix A = alloc_matrix(n);
    Matrix B = alloc_matrix(n);
    Matrix C = alloc_matrix(n);

    fill_random(&A);
    fill_random(&B);

    struct timespec t0, t1;

    if (modo_seq) {
        // ── Sequencial ──
        clock_gettime(CLOCK_MONOTONIC, &t0);
        matrix_multiply_seq(&A, &B, &C);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("Tempo sequencial : %.4f s\n", tempo);

    } else {
        // ── Paralelo ──
        clock_gettime(CLOCK_MONOTONIC, &t0);
        matrix_multiply_parallel(&A, &B, &C, num_threads);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("Tempo paralelo   : %.4f s\n", tempo);
        printf("\n(Para calcular speedup e eficiência, divida o tempo sequencial por %.4f)\n", tempo);
    }

    if (n <= 8) {
        printf("\nMatriz A:\n"); print_matrix(&A);
        printf("\nMatriz B:\n"); print_matrix(&B);
        printf("\nMatriz C:\n"); print_matrix(&C);
    }

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);

    return 0;
}