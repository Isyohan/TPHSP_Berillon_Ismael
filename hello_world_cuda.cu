#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void helloworld_cuda(){
    printf("Hello\n");
}

void MatrixMul(int *in1,int *in2 ,int a ,int *out ){
    for (int i=0;i<a*a;i++){
        int s=0;
        for (int j=0;j<a;j++){
            s+=in1[j+((int)i/a)*a]*in2[j*a+(i%a)];
             
        }
        out[i]=s;
    }
}

void MatrixInit(float *M, int n, int p){
	for(int i =0; i<n*p; i++){
		M[i] = (float)2*((float)rand()-((float)RAND_MAX)/2)/(float)RAND_MAX;
	} // [rand - (RAND_MAX/2)]/ (RAND_MAX/2) pour avoir un nombre aléatoire entre -1 et 1
}

void MatrixPrint(float *M, int n, int p){
    /* On stocke la matrice dans une liste de dimension 1 et de taille n*p:

    a b c d e f
    
    Cette liste désigne la matrice de taille n lignes et p colonnes :
    
    a b c 
    d e f

    L'indice i désigne les lignes
    L'indice j désigne les colonnes
    */
    for (int i=0;i<n;i++){
        for(int j=0;j<p;j++){
            printf("%1.2f\t",M[i*p+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i = 0; i<n*p;i++){
        Mout[i] = M1[i] + M2[i];
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i = blockIdx.x;
    int j = threadIdx.x;
//    printf("%d \t %d \n", i, j);

//    if(i*p+j>=n*p){printf("error");}
    Mout[i*p+j] = M1[i*p+j] + M2[i*p+j];
//    printf("%1.2f \n", Mout[i*p+j]);
}


int main(){
    helloworld_cuda<<<1,1>>>();
    srand( time(NULL) );
/*    int a=3;
    int in1[a*a];
    int in2[a*a];
    int out[a*a];
    for (int i=0; i<a*a;i++){
        in1[i]=i;
        in2[i]=i;
    }
    
    MatrixMul(in1,in2,a,out);
    for (int i=0;i<a*a;i++){
        printf("%d\n",out[i]);
    }*/
    int n = 3; // lignes
    int p = 2; // colonnes

    float* M1 = (float*) malloc(sizeof(float)*n*p);
    float* M2 = (float*) malloc(sizeof(float)*n*p);
    float* Mout = (float*) malloc(sizeof(float)*n*p);

    float* M1gpu;
    float* M2gpu;
    float* Moutgpu;
    (float*) cudaMalloc((void **) &M1gpu, sizeof(float)*n*p);
    (float*) cudaMalloc((void **) &M2gpu, sizeof(float)*n*p);
    (float*) cudaMalloc((void **) &Moutgpu, sizeof(float)*n*p);

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    cudaMemcpy(M1gpu,M1,sizeof(float)*n*p,cudaMemcpyHostToDevice);
    cudaMemcpy(M2gpu,M2,sizeof(float)*n*p,cudaMemcpyHostToDevice);
//    cudaMemcpy(Moutgpu,Mout,n*p,cudaMemcpyHostToDevice);
    cudaMatrixAdd<<<n,p>>>(M1gpu,M2gpu,Moutgpu,n,p);

 //   cudaMemcpy(M1,M1gpu,n*p,cudaMemcpyDeviceToHost);
 //   cudaMemcpy(M2,M2gpu,n*p,cudaMemcpyDeviceToHost);
    cudaMemcpy(Mout,Moutgpu,sizeof(float)*n*p,cudaMemcpyDeviceToHost);

    MatrixPrint(M1, n, p);
    MatrixPrint(M2, n, p);
    MatrixPrint(Mout, n, p);
    cudaDeviceSynchronize();
    printf("cuda\n");
    return 0;
}

