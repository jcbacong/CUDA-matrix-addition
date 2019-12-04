#include <stdio.h>
#include <stdlib.h>

void randomArray(int* cpu_array, unsigned long SIZE) {
	for(unsigned long i = 1; i<SIZE; i++){
		cpu_array[i] = rand()% + 1;
	}
}

int main() {
	unsigned long SIZE = 100;

	int* cpu_array = (int*) malloc(SIZE * sizeof(int));
	randomArray(cpu_array, SIZE);
	for(int i=0; i<SIZE; i++) {
		printf("Array index %d: %d\n", i, cpu_array[i]);
	}
	free(cpu_array);
	return 0;
}