#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "functions.cu"

int main(){

    srand( time(NULL) );


    int nin = 4; // dimensions matrice d'entrée
    int nout1 = 4; // dimensions matrice de sortie de la première couche convolutive
    int cout1 = 1; // Nombre de canaux de sortie de la première couche convolutive
    int nmaxpool = 2; // Dimensions de la matrice après max pooling
    int nkernel = 1; //Dimensions du noyau de convolution de la première couche convolutive

// Allocation de la mémoire
    float* raw_data = (float*) malloc(sizeof(float)*nin*nin);
    float* C1_data = (float*) malloc(sizeof(float)*nout1*nout1*cout1);
    float* S1_data = (float*) malloc(sizeof(float)*nmaxpool*nmaxpool*cout1);
    float* C1_kernel = (float*) malloc(sizeof(float)*nkernel*nkernel*cout1);
    float* Mout = (float*) malloc(sizeof(float)*nout1*nout1*cout1);
    float* Moutpool = (float*) malloc(sizeof(float)*nmaxpool*nmaxpool*cout1);
    
// Initialisation de la mémoire dans le gpu
    float* raw_data_gpu;
    float* C1_data_gpu;
    float* S1_data_gpu;
    float* C1_kernel_gpu;
    float* Mout_gpu;
    float* Moutpool_gpu;
    (float*) cudaMalloc((void **) &raw_data_gpu, sizeof(float)*nin*nin);
    (float*) cudaMalloc((void **) &C1_data_gpu, sizeof(float)*nout1*nout1*cout1);
    (float*) cudaMalloc((void **) &S1_data_gpu, sizeof(float)*nmaxpool*nmaxpool*cout1);
    (float*) cudaMalloc((void **) &C1_kernel_gpu, sizeof(float)*nkernel*nkernel*cout1);
    (float*) cudaMalloc((void **) &Mout_gpu, sizeof(float)*nout1*nout1*cout1);
    (float*) cudaMalloc((void **) &Moutpool_gpu, sizeof(float)*nmaxpool*nmaxpool*cout1);

//Initialisation des matrices
    MatrixInit(raw_data, nin, nin);
    MatrixInitChannel(C1_data, nout1, nout1, cout1);
    MatrixInitChannel(S1_data, nmaxpool, nmaxpool, cout1);
    MatrixInitChannel(C1_kernel, nkernel, nkernel, cout1);
    MatrixInitChannel(Mout, nout1, nout1, cout1);
    MatrixInitChannel(Moutpool,nmaxpool,nmaxpool,cout1);

//Copier la mémoire du CPU vers le GPU
    cudaMemcpy(raw_data_gpu,raw_data,sizeof(float)*nin*nin,cudaMemcpyHostToDevice);
    cudaMemcpy(C1_data_gpu,C1_data,sizeof(float)*nout1*nout1*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(S1_data_gpu,S1_data,sizeof(float)*nmaxpool*nmaxpool*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernel_gpu,C1_kernel,sizeof(float)*nkernel*nkernel*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(Mout_gpu,Mout,sizeof(float)*nout1*nout1*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(Moutpool_gpu,Moutpool,sizeof(float)*nmaxpool*nmaxpool*cout1,cudaMemcpyHostToDevice);    

    Conv2d<<<nout1,nout1>>>(raw_data_gpu, C1_kernel_gpu, Mout_gpu, nin, nkernel, 1, cout1); // Convolution
    MaxPoolingGlobal<<<nmaxpool,nmaxpool>>>(Mout_gpu, Moutpool_gpu, nmaxpool,2 , cout1); // Maxpooling
    
 
    cudaMemcpy(Mout,Mout_gpu,sizeof(float)*nout1*nout1*cout1,cudaMemcpyDeviceToHost); // Envoi de la matrice vers le cpu 
    cudaMemcpy(Moutpool,Moutpool_gpu,sizeof(float)*nmaxpool*nmaxpool*cout1,cudaMemcpyDeviceToHost);    


    printf("data : \n");
    MatrixPrint(raw_data,nin,nin);
    printf("\n");

    printf("kernels : \n");
    MatrixPrintChannel(C1_kernel, nkernel, nkernel,cout1);
    printf("\n");

    printf("Convolutionned : \n");
    MatrixPrintChannel(Mout, nout1, nout1,cout1); // Affichage du résultat de la convolution
    printf("\n");

    printf("Maxpooled : \n");
    MatrixPrintChannel(Moutpool, nmaxpool,nmaxpool,cout1); // Affichage du résultat après max pooling


    cudaDeviceSynchronize();


return 0;
}