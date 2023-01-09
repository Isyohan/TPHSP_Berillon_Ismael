#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test_fonc.cu"


int main(){
    srand( time(NULL) );

    int nin=32;
    int nb_channel_in=1;
    float* data = (float*) malloc(sizeof(float)*nin*nin*nb_channel_in);
    float* data_gpu;
    (float*) cudaMalloc((void **) &data_gpu, sizeof(float)*nin*nin*nb_channel_in);

    MatrixInitChannel(data, nin, nin,nb_channel_in);

    cudaMemcpy(data_gpu,data,sizeof(float)*nin*nin*nb_channel_in,cudaMemcpyHostToDevice);


    
    int nkernel=5;
    int ch_kernel=6;
    float* kernel = (float*) malloc(sizeof(float)*nkernel*nkernel*ch_kernel);
    float* kernel_gpu;
    (float*) cudaMalloc((void **) &kernel_gpu, sizeof(float)*nkernel*nkernel*ch_kernel);
    
    MatrixInitChannel(kernel, nkernel, nkernel,ch_kernel);
 

    cudaMemcpy(kernel_gpu,kernel,sizeof(float)*nkernel*nkernel*ch_kernel,cudaMemcpyHostToDevice);



    printf("kernel : \n");
    MatrixPrintChannel(kernel,nkernel,nkernel,ch_kernel);
    
    printf("data : \n");
    MatrixPrintChannel(data,nin,nin,nb_channel_in);


    int nout=nin-nkernel+1;
    
    float* out = (float*) malloc(sizeof(float)*nout*nout*ch_kernel);
    float* out_gpu;
    (float*) cudaMalloc((void **) &out_gpu, sizeof(float)*nout*nout*ch_kernel);




//    ConvNormal(data,kernel,out,nin,nkernel,nb_channel_in,ch_kernel);
    Conv2d<<<nout,nout>>>(data_gpu,kernel_gpu,out_gpu,nin,nkernel,nb_channel_in,ch_kernel);

    cudaMemcpy(out,out_gpu,sizeof(float)*nout*nout*ch_kernel,cudaMemcpyDeviceToHost); // Envoi de la matrice vers le cpu 

    printf("out:\n");
    MatrixPrintChannel(out,nout,nout,ch_kernel);

    return 0;
}