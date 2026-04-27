#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// Estrutura passada para cada thread 
typedef struct {
    double** A;
    double** B;
    double** C;
    int      n;
    int      row_start; // primeira linha responsabilidade desta thread
    int      row_end;   // última linha (exclusive)
} ThreadArgs;

// Alocação / liberação 
double** alloc_matrix(int n) {
    double** m = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
        m[i] = (double*)calloc(n, sizeof(double));
    return m;
}

void free_matrix(double** m, int n) {
    for (int i = 0; i < n; i++) free(m[i]);
    free(m);
}

void fill_random(double** m, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = rand() % 10;
}

void print_matrix(double** m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%6.1f ", m[i][j]);
        printf("\n");
    }
}

// Versão sequencial (referência) 
void matrix_multiply_seq(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// ─── Função executada por cada thread ────────────────────────────────────────
/*
 * Cada thread recebe um intervalo [row_start, row_end) de linhas da matriz C.
 * Como linhas distintas de C são completamente independentes entre si,
 * não há condição de corrida e nenhum mutex é necessário.
 *
 * Estratégia de cache: B é transposta antes da multiplicação (veja main),
 * então acessamos Bt[j][k] em vez de B[k][j] — ambos viram acessos de linha,
 * eliminando os cache misses na leitura de colunas de B.
 */
void* thread_worker(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    double**    A = a->A;
    double**    B = a->B; // na prática receberá B transposta
    double**    C = a->C;
    int         n = a->n;

    for (int i = a->row_start; i < a->row_end; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[j][k]; // B já está transposta → B[j][k]
            C[i][j] = sum;
        }
    }
    return NULL;
}

//  Transpõe B em Bt para melhorar localidade de cache 
double** transpose(double** B, int n) {
    double** Bt = alloc_matrix(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Bt[i][j] = B[j][i];
    return Bt;
}

//  Multiplicação paralela 
void matrix_multiply_parallel(double** A, double** B, double** C,
                               int n, int num_threads) {
    // Transpõe B uma única vez antes de distribuir o trabalho
    double** Bt = transpose(B, n);

    pthread_t*  threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs* args    = malloc(num_threads * sizeof(ThreadArgs));

    // Divide as linhas entre as threads (distribuição estática)
    int rows_per_thread = n / num_threads;
    int remainder       = n % num_threads;

    int row = 0;
    for (int t = 0; t < num_threads; t++) {
        // Threads iniciais recebem uma linha extra se n não for divisível
        int extra       = (t < remainder) ? 1 : 0;
        args[t].A       = A;
        args[t].B       = Bt;  // passa a transposta
        args[t].C       = C;
        args[t].n       = n;
        args[t].row_start = row;
        args[t].row_end   = row + rows_per_thread + extra;
        row = args[t].row_end;

        pthread_create(&threads[t], NULL, thread_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    free_matrix(Bt, n);
    free(threads);
    free(args);
}

int main(int argc, char* argv[]) {
    int n           = 512; // tamanho padrão
    int num_threads = 4;   // threads padrão

    if (argc > 1) n           = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);

    printf("=== Multiplicação de Matrizes com Threads ===\n");
    printf("Tamanho  : %d x %d\n", n, n);
    printf("Threads  : %d\n\n", num_threads);

    srand(42);

    double** A = alloc_matrix(n);
    double** B = alloc_matrix(n);
    double** C_seq = alloc_matrix(n);
    double** C_par = alloc_matrix(n);

    fill_random(A, n);
    fill_random(B, n);

    struct timespec t0, t1;

    // Sequencial 
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matrix_multiply_seq(A, B, C_seq, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tempo_seq = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Paralelo
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matrix_multiply_parallel(A, B, C_par, n, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tempo_par = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Resultados 
    printf("Tempo sequencial : %.4f s\n", tempo_seq);
    printf("Tempo paralelo   : %.4f s\n", tempo_par);
    printf("Speedup          : %.2fx\n\n", tempo_seq / tempo_par);

    // Verificação de corretude
    int ok = 1;
    for (int i = 0; i < n && ok; i++)
        for (int j = 0; j < n && ok; j++)
            if (C_seq[i][j] != C_par[i][j]) ok = 0;
    printf("Verificação      : %s\n", ok ? "OK — resultados idênticos ✓" : "ERRO — resultados divergem ✗");

    if (n <= 8) {
        printf("\nMatriz C (sequencial):\n"); print_matrix(C_seq, n);
        printf("\nMatriz C (paralela):\n");   print_matrix(C_par, n);
    }

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C_seq, n);
    free_matrix(C_par, n);

    return 0;
}