
#include <slib.h>

//NOTE(steven):Data[Layer][neuron]
//Weights[Layer][Neuron][WeightsForPrevLayerNeuron]

struct neural_network
{
    u32 InputCount;
    u32 OutputCount;
    u32 *LayerSizes;
    u32 LayerCount;
    u32 MaximumIterations;
    r32 Beta;
    r32 Alpha;
    r32 Epsilon;
    r32 *Data;
    u32 DataSize;
    r32 *Weights;
    u32 WeightSize;
};

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

internal r32
Sigmoid(r32 X)
{
    r32 Result = 1.0f/(1.0f + exp(-X));

    //Assert(Result < 1.0f && Result > 0);

    return Result;
}

internal void
SeedArrayRandomly(r32 *Array, u32 ArraySize)
{
    for(u32 ArrayIndex = 0;
        ArrayIndex < ArraySize;
        ++ArrayIndex)
    {
#if 1
        r32 RandomNumber = (r32)rand() / (RAND_MAX/2) - 1;
#else
        r32 RandomNumber = (r32)ArrayIndex;
#endif
        *(Array + ArrayIndex) = RandomNumber;
    }
}

internal void
PrintArray(r32 *Array, u32 ArraySize)
{
    printf("\n\nPrint Array\n\n");
    for(u32 ArrayIndex = 0;
        ArrayIndex < ArraySize;
        ++ArrayIndex)
    {
        printf("float %d: %f\n", ArrayIndex, *(Array + ArrayIndex));
    }
}

internal void
PrintNeuralNetwork(neural_network *NeuralNetwork)
{
    r32 *NeuronPointer = NeuralNetwork->Data;
    
    printf("\n\nNeural Network ---\n");
    for(u32 LayerIndex = 0;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        char *name;
        if(LayerIndex == 0)
        {
            name = "Input";
        }
        else if(LayerIndex == (NeuralNetwork->LayerCount - 1))
        {
            name = "Output";
        }
        else
        {
            name = "Hidden Layer";
        }

        for(u32 NeuronIndex = 0;
            NeuronIndex < *(NeuralNetwork->LayerSizes + LayerIndex);
            ++NeuronIndex)
        {
            r32 Value = *NeuronPointer++;
            printf("%s %d: %f\n", name, NeuronIndex, Value);
        }
    }
}

internal void
PrintWeights(neural_network *NeuralNetwork)
{
    u32 *LayerSizes = NeuralNetwork->LayerSizes;

    r32 *WeightPointer = NeuralNetwork->Weights;
    r32 *PrevNeuronPointer = NeuralNetwork->Data;

    r32 *LayerPointer = NeuralNetwork->Data + LayerSizes[0];
    
    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        u32 LayerSize = LayerSizes[LayerIndex];
        u32 PrevLayerSize = LayerSizes[LayerIndex - 1];

        r32 *NeuronPointer = LayerPointer;
        for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSize;
            ++NeuronIndex)
        {
            PrevNeuronPointer = LayerPointer - PrevLayerSize;

            for(u32 PrevLayerNeuronIndex = 0;
                PrevLayerNeuronIndex < PrevLayerSize;
                ++PrevLayerNeuronIndex)
            {
                r32 NeuronWeight = *WeightPointer++;
                printf("Weight from neuron %d in layer %d to neuron %d in next layer: %f\n",
                       PrevLayerNeuronIndex+1, LayerIndex, NeuronIndex+1, NeuronWeight);
            }

            r32 BiasWeight = *WeightPointer++;
            printf("bias to neuron %d in layer %d: %f\n", NeuronIndex+1, LayerIndex+1, BiasWeight);
            ++NeuronPointer;
        }
        
        LayerPointer += LayerSize;
    }    
}

internal void
InitializeNeuralNetwork(neural_network *NeuralNetwork)
{
    u32 DataArraySize = 0, WeightArraySize = 0, LayerSize = 0, PrevLayerSize = 0;
    for(u32 LayerIndex = 0;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        LayerSize = *(NeuralNetwork->LayerSizes + LayerIndex);
        DataArraySize += LayerSize;
        WeightArraySize += (PrevLayerSize) ? (PrevLayerSize + 1) * LayerSize : 0;

        PrevLayerSize = LayerSize;
    }

    NeuralNetwork->DataSize = DataArraySize;
    NeuralNetwork->WeightSize = WeightArraySize;

    NeuralNetwork->Data = (r32*)malloc(sizeof(r32) * DataArraySize);
    SeedArrayRandomly(NeuralNetwork->Data, DataArraySize);
    
    NeuralNetwork->Weights = (r32*)malloc(sizeof(r32) * WeightArraySize);
    SeedArrayRandomly(NeuralNetwork->Weights, WeightArraySize);
}

#define GetNeuralNetworkOutput(nn) (nn->Data + nn->DataSize - nn->OutputCount)

internal r32
MeanSquareError(neural_network *NeuralNetwork, r32 *DataPoint)
{
    r32 MSE = 0;

    r32 *Output = GetNeuralNetworkOutput(NeuralNetwork);

    for(u32 OutputIndex = 0;
        OutputIndex < NeuralNetwork->OutputCount;
        ++OutputIndex)
    {
        r32 TargetValue = *(DataPoint + OutputIndex);
        r32 OutputValue = *(Output + OutputIndex);
        r32 Delta = TargetValue - OutputValue;
        MSE += Delta * Delta;
    }

    return MSE * 0.5f;
}

internal void
FeedForward(neural_network *NeuralNetwork, r32 *DataPoint)
{
    u32 *LayerSizes = NeuralNetwork->LayerSizes;
    r32 *Data = NeuralNetwork->Data;
    r32 *Input = DataPoint;
    
    for(u32 InputIndex = 0;
        InputIndex < LayerSizes[0];
        ++InputIndex)
    {
        *Data++ = *Input++;
    }

    r32 *WeightPointer = NeuralNetwork->Weights;
    r32 *PrevNeuronPointer = NeuralNetwork->Data;

    r32 *LayerPointer = NeuralNetwork->Data + LayerSizes[0];

    //NOTE(steven): for each neuron, it's value is equal to the sum of the previous layer's
    //neuron values multiplied by their weight values, summed together (including a bias neuron).
    //This value then has the sigmoid function applied to it.
    
    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        u32 LayerSize = LayerSizes[LayerIndex];
        u32 PrevLayerSize = LayerSizes[LayerIndex - 1];

        r32 *NeuronPointer = LayerPointer;
        for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSize;
            ++NeuronIndex)
        {
            r32 Sum = 0;
            PrevNeuronPointer = LayerPointer - PrevLayerSize;

            for(u32 PrevLayerNeuronIndex = 0;
                PrevLayerNeuronIndex < PrevLayerSize;
                ++PrevLayerNeuronIndex)
            {
                r32 PrevNeuronValue = *PrevNeuronPointer++;
                r32 NeuronWeight = *WeightPointer++;
                Sum += PrevNeuronValue * NeuronWeight;
            }

            Sum += *WeightPointer++;
            *NeuronPointer++ = Sigmoid(Sum);
        }
        
        LayerPointer += LayerSize;
    }
}

internal void
BackPropogate(neural_network *NeuralNetwork, r32 *DataPoint)
{
    FeedForward(NeuralNetwork, DataPoint);
    
    u32 *LayerSizes = NeuralNetwork->LayerSizes;

    r32 *Delta = (r32 *)malloc((NeuralNetwork->DataSize - NeuralNetwork->InputCount) * sizeof(r32));
    r32 *Output = GetNeuralNetworkOutput(NeuralNetwork);

    r32 *DeltaPointer = Delta + (NeuralNetwork->DataSize -
                                 NeuralNetwork->InputCount -
                                 NeuralNetwork->OutputCount);

    // PrintNeuralNetwork(NeuralNetwork);
    
    //NOTE(steven): find delta for output, then hidden layers
    for(u32 OutputIndex = 0;
        OutputIndex < NeuralNetwork->OutputCount;
        ++OutputIndex)
    {
        DeltaPointer[OutputIndex] = DataPoint[OutputIndex] - Output[OutputIndex];
    }

    r32 *WeightPointer = NeuralNetwork->Weights + NeuralNetwork->WeightSize;

    r32 *NeuronPointer = Output;

    for(u32 HiddenLayerIndex = NeuralNetwork->LayerCount - 2;
        HiddenLayerIndex > 0;
        --HiddenLayerIndex)
    {
        NeuronPointer -= LayerSizes[HiddenLayerIndex];
        DeltaPointer -= LayerSizes[HiddenLayerIndex];
        
        for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSizes[HiddenLayerIndex];
            ++NeuronIndex)
        {
            WeightPointer -= (LayerSizes[HiddenLayerIndex] + 1);
            r32 Sum = 0.0f;
            for(u32 NextNeuronIndex = 0;
                NextNeuronIndex < LayerSizes[HiddenLayerIndex+1];
                ++NextNeuronIndex)
            {
                Sum += *(DeltaPointer + LayerSizes[HiddenLayerIndex] + NextNeuronIndex) *
                    *WeightPointer++;
            }

            r32 NeuronValue = *NeuronPointer++;
            *(DeltaPointer + NeuronIndex) = NeuronValue * (1 - NeuronValue) * Sum;
        }
    }

    //NOTE(steven): apply momentum - does nothing if (Alpha == 0)
    
    //NOTE(steven): adjust weights
    
    r32 *LayerPointer = NeuralNetwork->Data + LayerSizes[0];
    DeltaPointer = Delta;
    WeightPointer = NeuralNetwork->Weights;

    r32 *PrevNeuronPointer = NeuralNetwork->Data;
    
    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        u32 LayerSize = LayerSizes[LayerIndex];
        u32 PrevLayerSize = LayerSizes[LayerIndex - 1];

        r32 *NeuronPointer = LayerPointer;
        for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSize;
            ++NeuronIndex)
        {
            PrevNeuronPointer = LayerPointer - PrevLayerSize;
            
            for(u32 PrevLayerNeuronIndex = 0;
                PrevLayerNeuronIndex < PrevLayerSize;
                ++PrevLayerNeuronIndex)
            {
                r32 PrevNeuronValue = *PrevNeuronPointer++;
                r32 WeightAdjustment = (NeuralNetwork->Beta *
                                        *DeltaPointer *
                                        PrevNeuronValue);
                *WeightPointer++ += WeightAdjustment;
            }

            *WeightPointer++ += NeuralNetwork->Beta * *DeltaPointer++;
        }
        
        LayerPointer += LayerSize;
    }    

    free(Delta);
}

internal void
ShowResults(neural_network *NeuralNetwork, r32 *TestDataPoint, u32 TestIterationIndex)
{
    char *SuccessMsg = "Correct!";
    char *FailMsg = "Fail!";
    
    r32 *Output = GetNeuralNetworkOutput(NeuralNetwork);
    for(u32 OutputIndex = 0;
        OutputIndex < NeuralNetwork->OutputCount;
        ++OutputIndex)
    {
        r32 OutputValue = *(Output + OutputIndex);
        b32 Result = (b32)(OutputValue + 0.5f);
        u32 Answer = (u32)(*(TestDataPoint + NeuralNetwork->InputCount + OutputIndex) + 0.5f);
        char *Msg = (Result == Answer) ? SuccessMsg : FailMsg;
        printf("%s result: %d -- answer: %d\n\n", Msg, Result, Answer);
            
    }
}

internal void
FlatNN()
{
    srand(0);
    
    r32 TrainingData[] = {
        0,0,0,
        0,1,1,
        1,0,1,
        1,1,0,
    };
    
    neural_network NeuralNetwork = {};
    NeuralNetwork.InputCount = 2;
    NeuralNetwork.OutputCount = 1;
    
    u32 LayerSizes[] = {2,3,1};
    NeuralNetwork.LayerSizes = LayerSizes;
    NeuralNetwork.LayerCount = ArrayCount(LayerSizes);

    NeuralNetwork.Beta = 0.3f;
    NeuralNetwork.Alpha = 0.1f;
    NeuralNetwork.Epsilon = 0.0001f;

    NeuralNetwork.MaximumIterations = 500000;

    Assert(ArrayCount(TrainingData) % (NeuralNetwork.InputCount + NeuralNetwork.OutputCount) == 0);

    u32 TrainingDataPointCount = ArrayCount(TrainingData) / (NeuralNetwork.InputCount + NeuralNetwork.OutputCount);

    InitializeNeuralNetwork(&NeuralNetwork);

    for(u32 IterationIndex = 0;
        IterationIndex < NeuralNetwork.MaximumIterations;
        ++IterationIndex)
    {
        r32 *TrainDataPoint = TrainingData +
            ((IterationIndex % TrainingDataPointCount) *
             (NeuralNetwork.InputCount + NeuralNetwork.OutputCount));

        BackPropogate(&NeuralNetwork, TrainDataPoint);
        r32 MSE = MeanSquareError(&NeuralNetwork, TrainDataPoint);
        
        //if(MSE < NeuralNetwork.Epsilon) break;
    }

    for(u32 TestIterationIndex = 0;
        TestIterationIndex < TrainingDataPointCount;
        ++TestIterationIndex)
    {
        r32 *TestDataPoint = TrainingData +
            ((TestIterationIndex % TrainingDataPointCount) *
             NeuralNetwork.InputCount + NeuralNetwork.OutputCount);
        FeedForward(&NeuralNetwork, TestDataPoint);
        ShowResults(&NeuralNetwork, TestDataPoint, TestIterationIndex);
    }
}
