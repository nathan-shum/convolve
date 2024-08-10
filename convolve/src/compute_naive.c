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
  uint32_t counter = 0;

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

  for (int i = 0; i < pink_row; i++) {
    for (int j = 0; j < pink_col; j++) {
      uint32_t summed = 0;

      for (int k = 0; k < b_num_elms; k++) {
                if (k >= b_num_cols) {          
          uint32_t offset = (k / b_num_cols) * width_diff;
          summed += a_matrix->data[(i * a_num_cols) + j + k + offset] * flipped_b[k];
        } else {
          summed += a_matrix->data[(i * a_num_cols) + j + k] * flipped_b[k];
        }
      }
      output->data[counter] = summed;
      counter++;
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
