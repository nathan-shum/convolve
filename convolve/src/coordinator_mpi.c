#include <mpi.h>

#include "coordinator.h"

#define READY 0
#define NEW_TASK 1
#define TERMINATE -1

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Error: not enough arguments\n");
    printf("Usage: %s [path_to_task_list]\n", argv[0]);
    return -1;
  }

  // TODO: implement Open MPI coordinator
  // Based on discussion 10, question 4

  // Execute tasks
  // copied remaining structure from coordinator_naive.c
  // Read and parse task list file
  int numTasks = 0;
  task_t **tasks;
  read_tasks(argv[1], &numTasks, &tasks);
  MPI_Init(&argc, &argv); // initialize
  // get process ID of this process and total number of processes
  int procID, totalProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);
  // are we a manager or a worker?
  if (procID == 0) {
    int nextTask = 0; // next task to do
    MPI_Status status;
    int32_t message;
    // assign tasks
    while (nextTask < numTasks) {
      // wait for a message from any worker
      MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      int sourceProc = status.MPI_SOURCE; // process ID of the source of the message
      // assign next task
      message = nextTask;
      MPI_Send(&message, 1, MPI_INT32_T, sourceProc, 0, MPI_COMM_WORLD);
      nextTask++;
    }

    // wait for all processes to finish
    for (int i = 0; i < totalProcs - 1; i++) {
      // wait for a message from any worker
      MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      int sourceProc = status.MPI_SOURCE; // process ID of the source of the message
      message = TERMINATE;
      MPI_Send(&message, 1, MPI_INT32_T, sourceProc, 0, MPI_COMM_WORLD);
    }
  } else {
    int32_t message;
    while (true) {
      // request more work
      message = READY;
      MPI_Send(&message, 1, MPI_INT32_T, 0, 0, MPI_COMM_WORLD);

      // receive message from manager
      MPI_Recv(&message, 1, MPI_INT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (message == TERMINATE) break; // all done!

      execute_task(tasks[message]);
    } 
  }
  MPI_Finalize(); // clean up
}
