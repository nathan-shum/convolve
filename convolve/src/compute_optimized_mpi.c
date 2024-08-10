#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // TODO: convolve matrix a and matrix b, and store the resulting matrix in
  // output_matrix

  uint32_t a_num_rows = a_matrix->rows;
  uint32_t a_num_cols = a_matrix->cols;
  
  // Step 1: Flip matrix b by reversing original matrix b, stored in flipped_b
  uint32_t b_num_rows = b_matrix->rows;
  uint32_t b_num_cols = b_matrix->cols;
 
  uint32_t b_num_elms = b_num_cols * b_num_rows;
  uint32_t last_index = b_num_elms - 1;
  uint32_t flipped_b[b_num_elms];

  // according to ed; b matix can be simd-ed
  for (int i = last_index; i >= 0; i--) {
    flipped_b[i] = b_matrix->data[last_index - i];
  }

  // Step 2: Slide over matrix a with flipped b
  uint32_t height_diff = a_num_rows - b_num_rows;
  uint32_t width_diff = a_num_cols - b_num_cols;
  uint32_t pink_col = width_diff + 1;
  uint32_t pink_row = height_diff + 1;

  // area of pink matrix = col_pink * row_pink
  // col of pink matrix = difference(width_a, width_b) + 1
  // row of pink matrix = difference(height_a, height_b) + 1

  // allocate memory, check if failure return -1
  matrix_t* output = malloc(sizeof(matrix_t));
  if (output == NULL) {
    free(output);
    return -1;
  }
  output->data = malloc(sizeof(uint32_t) * pink_col * pink_row);
  if (output->data == NULL) {
    free(output->data);
    free(output);
    return -1;
  }
  // This pragma is faster but it messes up the internal calculations 
  // should technically reach desired speed up
  // no segfaults or memleaks; around ~148ms
  #pragma omp parallel for 
  for (uint32_t i = 0; i < pink_row; i++) {
    for (uint32_t j = 0; j < pink_col; j++) {
      // go through rows of b matrix/a submatrix one by one instead of trying to get the sub matrix from A all at once
      uint32_t summed = 0;
      // this is where we placed the pragma omp parallel in naive loop that worked
      // previously used #pragma omp paralle for reduction (: summed)
      for (uint32_t b_row = 0; b_row < b_num_rows; b_row++) {
        uint32_t temp = 0;
        // Initialize sum vector to {0, 0, 0, 0, 0, 0, 0, 0}
        __m256i sum_vec = _mm256_setzero_si256();
        uint32_t elts = b_num_cols / 8 * 8;
        for (uint32_t k = 0; k < elts; k += 8) {
          // Get correct rows from matrices A and B
          __m256i row_a = _mm256_loadu_si256((__m256i *) (a_matrix->data + k + j + (b_row * a_num_cols) + (i * a_num_cols)));
          __m256i row_b = _mm256_loadu_si256((__m256i *) (flipped_b + k + (b_row * b_num_cols)));
          // element wise multiplication of row a and b
          __m256i product = _mm256_mullo_epi32(row_a, row_b);
          // Add to existing sum vector
          sum_vec = _mm256_add_epi32(sum_vec, product);
        }
        // Tail case and also cases where b/bflipped num_cols < 8
        // index where leftover elms in row get calculated
        int left_over = elts;
        //#pragma omp parallel for 
        for (int k = left_over; k < b_num_cols; k++) {
          temp += a_matrix->data[j + k + (b_row * a_num_cols) + (i * a_num_cols)] * flipped_b[k + (b_row * b_num_cols)];
        }
        int arr[8];
        _mm256_storeu_si256(( __m256i  *) arr, sum_vec);
        summed += temp + arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
      }
      output->data[(i * pink_col) + j] = summed;
    }
  
  }
  *output_matrix = output;
  (*output_matrix)->data = output->data;
  (*output_matrix)->rows = pink_row;
  (*output_matrix)->cols = pink_col;

  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
