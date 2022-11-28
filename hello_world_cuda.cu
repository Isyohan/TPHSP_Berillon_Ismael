#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void helloworld_cuda(){
    printf("Hello\n");
}

void MatrixMul(float *in1,float *in2 ,int a ,float *out ){
    /*
    Matrice carrée d'ordre a
    On stocke la matrice dans une liste de dimension 1 et de taille n*p:

    a b c d 

    Cette liste désigne la matrice de taille n lignes et p colonnes :
    
    a b 
    c d

    e f
    g h

    Résultat :
    a*e+b*g a*f+b*h
    c*e+d*g c*f+d*h
    
    Indice i désigne les lignes
    Indice j déqigne les colonnes
    */
    for (int i=0;i<a*a;i++){
        int s=0;
        for (int j=0;j<a;j++){
            s+=in1[j+((int)i/a)*a]*in2[j*a+(i%a)];
             
        }
        out[i]=s;
    }
}

__global__ void cudaMatrixMul(float *M1, float*M2, float *Mout, int n){
    int i = blockIdx.x;
    int j = threadIdx.x;
    float temp = 0;

    for(int k = 0; k<n; k++){
        temp+= M1[k+i*n]*M2[n*k+j]; //i : lignes, j : colonnes
    }
    Mout[i*n+j]=temp;
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
    int n = 2'000; // lignes
 //   int p = 2; // colonnes

    float* M1 = (float*) malloc(sizeof(float)*n*n);
    float* M2 = (float*) malloc(sizeof(float)*n*n);
    float* Mout = (float*) malloc(sizeof(float)*n*n);

    float* M1gpu;
    float* M2gpu;
    float* Moutgpu;
    (float*) cudaMalloc((void **) &M1gpu, sizeof(float)*n*n);
    (float*) cudaMalloc((void **) &M2gpu, sizeof(float)*n*n);
    (float*) cudaMalloc((void **) &Moutgpu, sizeof(float)*n*n);

    MatrixInit(M1, n, n);
    MatrixInit(M2, n, n);

    //MatrixMul(M1,M2,n,Mout);

    
    cudaMemcpy(M1gpu,M1,sizeof(float)*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(M2gpu,M2,sizeof(float)*n*n,cudaMemcpyHostToDevice);

    cudaMatrixMul<<<n,n>>>(M1gpu,M2gpu,Moutgpu,n);

    cudaMemcpy(Mout,Moutgpu,sizeof(float)*n*n,cudaMemcpyDeviceToHost);
    
    

    //MatrixPrint(M1, n, n);
    //MatrixPrint(M2, n, n);
    //MatrixPrint(Mout, n, n);
    cudaDeviceSynchronize();
    printf("n = %d\n",n);
    return 0;
}

