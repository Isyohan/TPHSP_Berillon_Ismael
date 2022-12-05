#include "hello_world_cuda.cu"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    
    int nin = 32; // dimensions matrice d'entrée
    int nout1 = 28; // dimensions matrice de sortie de la première couche convolutive
    int cout1 = 6; // Nombre de canaux de sortie de la première couche convolutive
    int nmaxpool = 14; // Dimensions de la matrice après max pooling
    int nkernel = 5; //Dimensions du noyau de convolution de la première couche convolutive

// Allocation de la mémoire
    float* raw_data = (float*) malloc(sizeof(float)*nin*nin);
    float* C1_data = (float*) malloc(sizeof(float)*nout1*nout1*cout1);
    float* S1_data = (float*) malloc(sizeof(float)*nmaxpool*nmaxpool*cout1);
    float* C1_kernel = (float*) malloc(sizeof(float)*nkernel*nkernel*cout1);

// Initialisation de la mémoire dans le gpu
    float* raw_data_gpu;
    float* C1_data_gpu;
    float* S1_data_gpu;
    float* C1_kernel_gpu;
    (float*) cudaMalloc((void **) &raw_data_gpu, sizeof(float)*nin*nin);
    (float*) cudaMalloc((void **) &C1_data_gpu, sizeof(float)*nout1*nout1*cout1);
    (float*) cudaMalloc((void **) &S1_data_gpu, sizeof(float)*nmaxpool*nmaxpool*cout1);
    (float*) cudaMalloc((void **) &C1_kernel_gpu, sizeof(float)*nkernel*nkernel*cout1);

//Initialisation des matrices
    MatrixInit(raw_data, nin, nin);
    MatrixInitChannel(C1_data, nout1, nout1, cout1);
    MatrixInitChannel(S1_data, nmaxpool, nmaxpool, cout1);
    MatrixInitChannel(C1_kernel, nkernel, nkernel, cout1);

//Copier la mémoire du CPU vers le GPU
    cudaMemcpy(raw_data_gpu,raw_data,sizeof(float)*nin*nin,cudaMemcpyHostToDevice);
    cudaMemcpy(C1_data_gpu,C1_data,sizeof(float)*nout1*nout1*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(S1_data_gpu,S1_data,sizeof(float)*nmaxpool*nmaxpool*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernel_gpu,C1_kernel,sizeof(float)*nkernel*nkernel*cout1,cudaMemcpyHostToDevice);



return 0;
}