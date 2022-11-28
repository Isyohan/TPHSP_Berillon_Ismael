#include <stdio.h>

__global__ void helloworld_cuda(){
    printf("Hello\n");
}

void mulmat(int *in1,int *in2 ,int a ,int *out ){
    for (int i=0;i<a*a;i++){
        int s=0;
        for (int j=0;j<a;j++){
            s+=in1[j+((int)i/a)*a]*in2[j*a+(i%a)];
             
        }
        out[i]=s;
    }
}

int main(){
    helloworld_cuda<<<1,1>>>();
    int a=3;
    int in1[a*a];
    int in2[a*a];
    int out[a*a];
    for (int i=0; i<a*a;i++){
        in1[i]=i;
        in2[i]=i;
    }
    
    mulmat(in1,in2,a,out);
    for (int i=0;i<a*a;i++){
        printf("%d\n",out[i]);
    }

    cudaDeviceSynchronize();
    printf("cuda\n");
    return 0;
}

