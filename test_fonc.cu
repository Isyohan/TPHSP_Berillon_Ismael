#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void MatrixPrintChannel(float *M,int n,int p ,int c){
    for (int ch=0;ch<c;ch++){
        for (int i=0;i<n;i++){
            for(int j=0;j<p;j++){
                printf("%1.2f\t",M[i*p+j+n*p*ch]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}




__device__ float MatrixMulTermToTerm(float *M1, float *M2, int n){ // Pour faire la convolution
    float sum = 0;
    for(int i = 0; i<n*n;i++){
        sum += M1[i]*M2[i];
    }
    return sum;
}

float MatrixMulTermToTermNormal(float *M1, float *M2, int n){ // Pour faire la convolution
    float sum = 0;
    for(int i = 0; i<n*n;i++){
        sum += M1[i]*M2[i];
    }
    return sum;
}

__device__ void ChooseChannel(float *M1, float *Mout, int n, int c){
    for(int i=0; i<n*n; i++){
        Mout[i] = M1[i+c*n*n];
    }

}

void ChooseChannelNormal(float *M1, float *Mout, int n, int c){
    for(int i=0; i<n*n; i++){
        Mout[i] = M1[i+c*n*n];
    }

}

void SubMatrixNormal(float *M1, float *Mout,int nin, int n,int c, int i, int j){ // Récupérer la matrice de taille n*n à partir de l'indice (i,j)
    for (int ch=0;ch<c;ch++){
        for(int k = 0; k<n; k++){
            for(int l = 0; l < n; l++){
                Mout[k + l*n + n*n*ch] = M1[k+j + (l+i)*nin + nin*nin*ch];
            }
        }
    }
}
__device__ void SubMatrixDevice(float *M1, float *Mout,int nin, int n,int c, int i, int j){ // Récupérer la matrice de taille n*n à partir de l'indice (i,j)
    for (int ch=0;ch<c;ch++){
        for(int k = 0; k<n; k++){
            for(int l = 0; l < n; l++){
                Mout[k + l*n + n*n*ch] = M1[k+j + (l+i)*nin + nin*nin*ch];
            }
        }
    }
}

void ConvNormal(float* Min ,float* kernels ,float* Mout ,int nin ,int nkernel ,int channel_in ,int channel_kernel){
    int nout=nin-nkernel+1;
    float* subM = (float*) malloc(sizeof(float)*nkernel*nkernel);
    float* oneChannelKernel = (float*) malloc(sizeof(float)*nkernel*nkernel);
    for (int i=0;i<nout;i++){
        for (int j=0;j<nout;j++){
            SubMatrixNormal(Min,subM,nin,nkernel,channel_in,i,j);
            for (int ch=0 ; ch<channel_kernel ; ch++){
                ChooseChannelNormal(kernels,oneChannelKernel,nkernel,ch);
                Mout[i*nout + j + ch*nout*nout]=MatrixMulTermToTermNormal(subM,oneChannelKernel,nkernel);
            }
        }
    }
    
}

__global__ void Conv2d(float* Min ,float* kernels ,float* Mout ,int nin ,int nkernel ,int channel_in ,int channel_kernel){
    int nout=nin-nkernel+1;
    float* subM = (float*) malloc(sizeof(float)*nkernel*nkernel);
    float* oneChannelKernel = (float*) malloc(sizeof(float)*nkernel*nkernel);

    int j = blockIdx.x;
    int i = threadIdx.x;


    SubMatrixDevice(Min,subM,nin,nkernel,channel_in,i,j);
    for (int ch=0 ; ch<channel_kernel ; ch++){
        ChooseChannel(kernels,oneChannelKernel,nkernel,ch);
        Mout[i*nout + j + ch*nout*nout]=MatrixMulTermToTerm(subM,oneChannelKernel,nkernel);
    }
}


float MaxMatNormal(float *F, int red){
    float max = -1.0;

    for(int i = 0; i < red*red; i++){
        if(max<F[i]){
            max = F[i];
        }
    }
    return max;
}

void maxpoolNormal(float* Min, float* Mout, int nout, int taille_maxpooling, int n_channel){
    float* subM = (float*) malloc(sizeof(float)*taille_maxpooling*taille_maxpooling*n_channel);
    float* oneChannelMaxpooling = (float*) malloc(sizeof(float)*taille_maxpooling*taille_maxpooling);


    for(int i=0;i<=nout;i+=1){
        for(int j=0;j<=nout;j+=1){
            SubMatrixNormal(Min,subM,nout*taille_maxpooling,taille_maxpooling,n_channel,i*taille_maxpooling,j*taille_maxpooling);

            for (int ch=0;ch<n_channel;ch++){
                ChooseChannelNormal(subM,oneChannelMaxpooling,taille_maxpooling,ch);

                //printf("(%d,%d,%d)\n",i,j,ch);
                //MatrixPrintChannel(oneChannelMaxpooling,taille_maxpooling,taille_maxpooling,1);

                Mout[j + i*nout + ch*nout*nout]=MaxMatNormal(oneChannelMaxpooling,taille_maxpooling);
            }
        }
    }
}


__device__ float MaxMatDevice(float *F, int red){
    float max = -1.0;

    for(int i = 0; i < red*red; i++){
        if(max<F[i]){
            max = F[i];
        }
    }
    return max;
}

__global__ void MaxPoolingGlobal(float* Min, float* Mout, int nout, int taille_maxpooling, int n_channel){
    float* subM = (float*) malloc(sizeof(float)*taille_maxpooling*taille_maxpooling*n_channel);
    float* oneChannelMaxpooling = (float*) malloc(sizeof(float)*taille_maxpooling*taille_maxpooling);


    int j = blockIdx.x;
    int i = threadIdx.x;

    SubMatrixDevice(Min,subM,nout*taille_maxpooling,taille_maxpooling,n_channel,i*taille_maxpooling,j*taille_maxpooling);

    for (int ch=0;ch<n_channel;ch++){
        ChooseChannel(subM,oneChannelMaxpooling,taille_maxpooling,ch);

                //printf("(%d,%d,%d)\n",i,j,ch);
                //MatrixPrintChannel(oneChannelMaxpooling,taille_maxpooling,taille_maxpooling,1);

        Mout[j + i*nout + ch*nout*nout]=MaxMatDevice(oneChannelMaxpooling,taille_maxpooling);
    }
       
}
