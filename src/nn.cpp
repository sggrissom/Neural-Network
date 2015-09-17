
#include <slib.h>

#if FLAT_NN
#include "FlatNN.h"
#elif SIMPLE_NN
#include "SimpleNN.h"
#endif

s32 main()
{
    srand(0);
    
    r32 TrainingData[] = {
        0,0,0,0,
        0,0,1,1,
        0,1,0,1,
        0,1,1,0,
        1,0,0,1,
        1,0,1,0,
        1,1,0,0,
        1,1,1,1,
    };
    
    neural_network NeuralNetwork = {};
    NeuralNetwork.InputCount = 3;
    NeuralNetwork.OutputCount = 1;
    
    u32 LayerSizes[] = {3,3,2,1};
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

    PrintWeights(&NeuralNetwork);


    char *SuccessMsg = "Correct!";
    char *FailMsg = "Fail!";
    
    for(u32 TestIterationIndex = 0;
        TestIterationIndex < TrainingDataPointCount;
        ++TestIterationIndex)
    {
        r32 *TestDataPoint = TrainingData +
            ((TestIterationIndex % TrainingDataPointCount) *
             NeuralNetwork.InputCount + NeuralNetwork.OutputCount);
        u32 Answer = (u32)(*(TestDataPoint + NeuralNetwork.InputCount) + 0.5f);
        FeedForward(&NeuralNetwork, TestDataPoint);
        r32 *Output = NeuralNetwork.Data + NeuralNetwork.DataSize - NeuralNetwork.OutputCount;
        for(u32 OutputIndex = 0;
            OutputIndex < NeuralNetwork.OutputCount;
            ++OutputIndex)
        {
            r32 OutputValue = *(Output + OutputIndex);
            PrintArray(TestDataPoint, NeuralNetwork.InputCount);
            b32 Result = (b32)(OutputValue + 0.5f);
            char *Msg = (Result == Answer) ? SuccessMsg : FailMsg;
            printf("%s result: %d -- answer: %d\n\n", Msg, Result, Answer);
            
        }
    }
    
    return 1;
}
