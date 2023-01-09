#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

void MatrixInitChannel(float *M, int n, int p, int c){
	for(int i =0; i<n*p*c; i++){
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

__device__ float MatrixMulTermToTerm(float *M1, float *M2, int n){ // Pour faire la convolution
    float sum = 0;
    for(int i = 0; i<n*n;i++){
        sum += M1[i]*M2[i];
    }
    return sum;
}

__device__ void SubMatrix(float *M1, float *Mout, int n, int i, int j,int c,int N){ // Récupérer la matrice de taille 5*5 à partir de l'indice (i,j)

    for(int k = 0; k<n; k++){
        for(int l = 0; l < n; l++){
            Mout[l+k*n] = M1[l+j+(k+i)*N+c*N*N];
        }
    }
}

__device__ void ChooseChannel(float *M1, float *Mout, int n, int c){
    for(int i=0; i<n*n; i++){
        Mout[i] = M1[i+c*n*n];
    }

}

__global__ void cudaConvolutionMatrix(float *M1, float *M2, float *Mout, int n, int k, int c){ // Réalisation de la convolution
    int i = blockIdx.x;
    int j = threadIdx.x;
    float* M = (float*) malloc(sizeof(float)*k*k); // Sous matrice locale pour la convolution
    float* F = (float*) malloc(sizeof(float)*k*k); // Sous matrice pour chaque canal d'entrée

    //SubMatrix(M1, M, k, i, j,n+k-1);
    for(int ch = 0; ch < c; ch++){ // Pour chaque canal
        SubMatrix(M1, M, k, i, j,ch,n+k-1);
        ChooseChannel(M2, F, k, ch);
        Mout[i*n+j+ch*n*n] = MatrixMulTermToTerm(F,M,k); // Convolution pour chaque canal
    }

}

__device__ float MaxMat(float *F, int red){
    float max = -1.0;

    for(int i = 0; i < red*red; i++){
        if(max<F[i]){
            max = F[i];
        }
    }
    return max;
}

__global__ void cudaMaxPooling(float *M1, float *Mout, int red, int nout, int c){
    int i = blockIdx.x;
    int j = threadIdx.x;
    printf("i=%d , j=%d\n",i,j);
    int nin = nout*red;
    float* F = (float*) malloc(sizeof(float)*red*red); // Sous matrice pour chaque canal dans laquelle on va choisir le maximum
    float* SubM = (float*) malloc(sizeof(float)*red*red*c); // Sous matrice de taille 2*2*6

    //SubMatrix(M1, SubM, red, red*i, red*j, nin); // red désigne le paramètre par lequel on va réduire la matrice, ici red = 2

    for(int ch = 0; ch < c; ch++){
        SubMatrix(M1, SubM, red, (red+1)*i, (red+1)*j, ch, nin); // red désigne le paramètre par lequel on va réduire la matrice, ici red = 2
        ChooseChannel(SubM, F, red, ch);
        Mout[i*nout+j+c*nout*nout] =  MaxMat(F,red); // On choisit le maximum de la matrice F 
        Mout[0]=MaxMat(F,red);
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


int mainTP1(){
    
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

