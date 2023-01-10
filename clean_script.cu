#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "functions.cu"

int main(){

    srand( time(NULL) );


    int nin = 32; // dimensions matrice d'entrée
    int cin = 1;

    int nout1 = 28; // dimensions matrice de sortie de la première couche convolutive
    int cout1 = 6; // Nombre de canaux de sortie de la première couche convolutive
    
    int nmaxpool = 14; // Dimensions de la matrice après mean pooling

    int nout2 = 10 // dimensions matrice de sortie de la second couche convolutive
    int cout2 = 16; // Nombre de canaux de sortie de la seconde couche convolutive

    int nmeanpool2 = 5; // Dimensions de la matrice après second mean pooling

    int nkernel = 5; //Dimensions du noyau de convolution de la première & seconde couche convolutive

    int nd_1 = 400; // Dimension du vecteur de la matrice de sortie applatie = 5*5*16
    int nd_2 = 120; // Dimension de sortie de la première couche linéaire
    int nd_3 = 84; // Dimension de sortie de la deuxième couche linéaire
    int nd_4 = 10; // Dimension de sortie de la troisième couche linéaire

// Allocation de la mémoire
    float* raw_data = (float*) malloc(sizeof(float)*nin*nin*cin);

    float* C1_kernel = (float*) malloc(sizeof(float)*nkernel*nkernel*cout1);
    float* biais1 = (float*) malloc(sizeof(float)*cout1);

    float* Mout = (float*) malloc(sizeof(float)*nout1*nout1*cout1);
    float* Moutpool = (float*) malloc(sizeof(float)*nmaxpool*nmaxpool*cout1);

    float* C2_kernel = (float*) malloc(sizeof(float)*nkernel*nkernel*cout1*cout2);
    float* biais2 = (float*) malloc(sizeof(float)*cout2);

    float* Mout2=(float*) malloc(sizeof(float)*nout2*nout2*cout2);
    float* Moutpool2 = (float*) malloc(sizeof(float)*nmeanpool2*nmeanpool2*cout2);


    float* M_dense1=(float*) malloc(sizeof(float)*nd_2*nd_1);
    float* V_out1= (float*) malloc(sizeof(float)*nd_2);
    float* biais3 = (float*) malloc(sizeof(float)*nd_2);

    float* M_dense2=(float*) malloc(sizeof(float)*nd_3*nd_2);
    float* V_out2= (float*) malloc(sizeof(float)*nd_3);
    float* biais4 = (float*) malloc(sizeof(float)*nd_3);

    float* M_dense3=(float*) malloc(sizeof(float)*nd_4*nd_3);
    float* V_out3= (float*) malloc(sizeof(float)*nd_4);
    float* biais5 = (float*) malloc(sizeof(float)*nd_4);
    

// Initialisation de la mémoire dans le gpu


    float* raw_data_gpu; (float*) cudaMalloc((void **) &raw_data_gpu, sizeof(float)*nin*nin);

    float* C1_kernel_gpu; (float*) cudaMalloc((void **) &C1_kernel_gpu, sizeof(float)*nkernel*nkernel*cout1);
    float* biais1_gpu; (float*) cudaMalloc((void **) &biais1_gpu, sizeof(float)*cout1);

    float* Mout_gpu; (float*) cudaMalloc((void **) &Mout_gpu, sizeof(float)*nout1*nout1*cout1);
    float* Moutpool_gpu; (float*) cudaMalloc((void **) &Moutpool_gpu, sizeof(float)*nmaxpool*nmaxpool*cout1);

    float* C2_kernel_gpu; (float*) cudaMalloc((void **) &C2_kernel_gpu, sizeof(float)*nkernel*nkernel*cout1*cout2)
    float* biais2_gpu; (float*) cudaMalloc((void **) &biais1_gpu, sizeof(float)*cout2);

    float* Mout2_gpu; (float*) cudaMalloc((void **) &Mout2_gpu, sizeof(float)*nout2*nout2*cout2);
    float* Moutpool2_gpu; (float*) cudaMalloc((void **) &Moutpool2_gpu, sizeof(float)*nmeanpool2*nmeanpool2*cout2);


    float* M_dense1_gpu; (float*) cudaMalloc((void **) &M_dense1_gpu, sizeof(float)*nd_2*nd_1);
    float* V_out1_gpu;
    float* biais3_gpu;

    float* M_dense2_gpu; (float*) cudaMalloc((void **) &M_dense2_gpu, sizeof(float)*nd_3*nd_2);
    float* V_out2_gpu;
    float* biais4_gpu;

    float* M_dense3_gpu; (float*) cudaMalloc((void **) &M_dense3_gpu, sizeof(float)*nd_4*nd_3);
    float* V_out3_gpu;
    float* biais5_gpu;

    (float*) cudaMalloc((void **) &raw_data_gpu, sizeof(float)*nin*nin);
    (float*) cudaMalloc((void **) &C1_kernel_gpu, sizeof(float)*nkernel*nkernel*cout1);
    (float*) cudaMalloc((void **) &Mout_gpu, sizeof(float)*nout1*nout1*cout1);
    (float*) cudaMalloc((void **) &Moutpool_gpu, sizeof(float)*nmaxpool*nmaxpool*cout1);

    (float*) cudaMalloc((void **) &biais1_gpu, sizeof(float)*cout1);
    (float*) cudaMalloc((void **) &biais2_gpu, sizeof(float)*cout2);
    (float*) cudaMalloc((void **) &biais3_gpu, sizeof(float)*nd_2);
    (float*) cudaMalloc((void **) &biais4_gpu, sizeof(float)*nd_3);
    (float*) cudaMalloc((void **) &biais5_gpu, sizeof(float)*nd_4);

//Initialisation des matrices
    MatrixInit(raw_data, nin, nin);

    MatrixInitChannel(C1_kernel, nkernel, nkernel, cout1);
    //MatrixInitChannel(Mout, nout1, nout1, cout1);
    //MatrixInitChannel(Moutpool,nmaxpool,nmaxpool,cout1);
    MatrixInit(biais1, cout1, 1);

    MatrixInitChannel(C2_kernel, nkernel, nkernel, cout1*cout2);
    MatrixInit(biais2, cout2, 1);






    //biais1[0]=0;

//Copier la mémoire du CPU vers le GPU
    cudaMemcpy(raw_data_gpu,raw_data,sizeof(float)*nin*nin,cudaMemcpyHostToDevice);
    cudaMemcpy(C1_data_gpu,C1_data,sizeof(float)*nout1*nout1*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(S1_data_gpu,S1_data,sizeof(float)*nmaxpool*nmaxpool*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernel_gpu,C1_kernel,sizeof(float)*nkernel*nkernel*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(Mout_gpu,Mout,sizeof(float)*nout1*nout1*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(Moutpool_gpu,Moutpool,sizeof(float)*nmaxpool*nmaxpool*cout1,cudaMemcpyHostToDevice);
    cudaMemcpy(biais1_gpu,biais1,sizeof(float)*cout1,cudaMemcpyHostToDevice);

    //Conv2d<<<nout1,nout1>>>(raw_data_gpu, C1_kernel_gpu, Mout_gpu, nin, nkernel, 1, cout1, biais1_gpu); // Convolution
    Conv2d_multi_channel_in<<<nout1,nout1>>>(raw_data_gpu, C1_kernel_gpu, Mout_gpu, nin, nkernel, 1, cout1, biais1_gpu);
    AveragePoolingGlobal<<<nmaxpool,nmaxpool>>>(Mout_gpu, Moutpool_gpu, nmaxpool,2 , cout1); // Maxpooling
    
 
    cudaMemcpy(Mout,Mout_gpu,sizeof(float)*nout1*nout1*cout1,cudaMemcpyDeviceToHost); // Envoi de la matrice vers le cpu 
    cudaMemcpy(Moutpool,Moutpool_gpu,sizeof(float)*nmaxpool*nmaxpool*cout1,cudaMemcpyDeviceToHost);    


    printf("data : \n");
    MatrixPrint(raw_data,nin,nin);
    printf("\n");

    printf("kernels : \n");
    MatrixPrintChannel(C1_kernel, nkernel, nkernel,cout1);
    printf("\n");

    printf("biais : \n");
    MatrixPrintChannel(biais1, cout1, 1,1);
    printf("\n");

    printf("Convolutioned : \n");
    MatrixPrintChannel(Mout, nout1, nout1,cout1); // Affichage du résultat de la convolution
    printf("\n");

    printf("Maxpooled : \n");
    MatrixPrintChannel(Moutpool, nmaxpool,nmaxpool,cout1); // Affichage du résultat après max pooling

/*
    int n_in = 2;
    int n_out = 1;
    float* V_in= (float*) malloc(sizeof(float)*n_in);
    float* V_out = (float*) malloc(sizeof(float)*n_out);
    float* biais = (float*) malloc(sizeof(float)*n_out);
    float* M_poids = (float*) malloc(sizeof(float)*n_in*n_out);

    float* V_in_gpu;
    float* V_out_gpu;
    float* biais_gpu;
    float* M_poids_gpu;

    (float*) cudaMalloc((void **) &V_in_gpu, sizeof(float)*n_in);
    (float*) cudaMalloc((void **) &V_out_gpu, sizeof(float)*n_out);
    (float*) cudaMalloc((void **) &biais_gpu, sizeof(float)*n_out);
    (float*) cudaMalloc((void **) &M_poids_gpu, sizeof(float)*n_in*n_out);

    MatrixInit(V_in, n_in, 1);
    MatrixInit(biais, n_out, 1);
    MatrixInit(M_poids, n_out, n_in);

    cudaMemcpy(V_in_gpu,V_in,sizeof(float)*n_in,cudaMemcpyHostToDevice);
    cudaMemcpy(biais_gpu,biais,sizeof(float)*n_out,cudaMemcpyHostToDevice);
    cudaMemcpy(M_poids_gpu,M_poids,sizeof(float)*n_in*n_out,cudaMemcpyHostToDevice);

    Dense<<<1,n_out>>>(V_in_gpu, V_out_gpu, M_poids_gpu, biais_gpu, n_in, n_out);
    //DenseNormal(V_in, V_out, M_poids, biais, n_in, n_out);
    cudaMemcpy(V_out,V_out_gpu,sizeof(float)*n_out,cudaMemcpyDeviceToHost);   

    printf("Test Dense : \n");
    printf("V_in : \n");
    MatrixPrint(V_in, n_in, 1);
    printf("biais : \n");
    MatrixPrint(biais, n_out, 1);
    printf("M_poids : \n");
    MatrixPrint(M_poids, n_out, n_in);
    printf("V_out : \n");
    MatrixPrint(V_out, n_out, 1);*/

    cudaDeviceSynchronize();


return 0;
}