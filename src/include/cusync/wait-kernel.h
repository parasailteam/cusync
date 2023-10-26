#include "device-functions.h"

#pragma once

namespace cusync {
/*
 * The wait kernel waits until the value of semaphore has reached the given value.
 * @semaphore: Address to the unsigned integer semaphore 
 * @givenValue: Given value of the semaphore 
*/
CUSYNC_GLOBAL
void waitKernel(volatile uint* semaphore, uint givenValue) {
  if (threadIdx.x == 0) {
    uint currVal = globalLoad(semaphore);
    while(currVal < givenValue) {
      currVal = globalVolatileLoad(semaphore);
    }
  }
}
}