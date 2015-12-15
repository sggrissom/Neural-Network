
#include <time.h>
#include <math.h>

#include <slib.h>

#include "NeuralNetwork.cpp"

#if FLAT
#include "FlatNN.cpp"
#elif UNROLLED
#include "UnrolledNN.cpp"
#endif

s32 main()
{
    srand(0);
    
    neural_network NeuralNetwork = {};
    
    u32 LayerSizes[] = LAYERSIZES;
    NeuralNetwork.LayerSizes = LayerSizes;
    NeuralNetwork.LayerCount = ArrayCount(LayerSizes);
    NeuralNetwork.InputCount = LayerSizes[0];
    NeuralNetwork.OutputCount = LayerSizes[NeuralNetwork.LayerCount - 1];
    NeuralNetwork.Beta = BETA;
    NeuralNetwork.Alpha = ALPHA;
    NeuralNetwork.Epsilon = EPSILON;

    u32 MaximumIterations =  MAX_ITERATIONS;

    u32 InputCount = NeuralNetwork.InputCount;
    u32 OutputCount = NeuralNetwork.OutputCount;
    u32 LayerCount = NeuralNetwork.LayerCount;


#if IRIS || DIGITS
    r32 *TrainingData = 0;
    u32 DataCount = LoadCSV(FILENAME, &TrainingData);
    Assert(DataCount % (InputCount + OutputCount) == 0);
#else
    u32 DataCount = ArrayCount(TrainingData);
    Assert(DataCount % (InputCount + OutputCount) == 0);
#endif

    u32 TrainingDataPointCount = DataCount / (InputCount + OutputCount);
    u32 TrainDataPitch = InputCount + OutputCount;

    InitializeNetwork(&NeuralNetwork);

    clock_t startTime = clock();
    
    for(u32 IterationIndex = 0;
        IterationIndex < MaximumIterations*TrainDataPitch;
        ++IterationIndex)
    {
        r32 *DataPoint = &TrainingData[(IterationIndex%TrainingDataPointCount)*TrainDataPitch];
        r32 *Target = DataPoint + InputCount;
        BackPropogate(&NeuralNetwork, DataPoint, Target);
    }

    clock_t endTime = clock();
    clock_t clockTicksTaken = endTime - startTime;
    r32 timeInSeconds = clockTicksTaken / (r32) CLOCKS_PER_SEC;
    
    printf("%f s\n", timeInSeconds);

#if 0
    for(u32 i=0;
        i < TrainingDataPointCount;
        ++i)
	{
		FeedForward(&NeuralNetwork,
                    &TrainingData[i*TrainDataPitch]);
		for(u32 j=0;
             j < InputCount;
             ++j)
		{
			r32 Value = TrainingData[i*(TrainDataPitch)+j];
            printf("%f ", Value);
		}

        for(u32 j=0;
            j < OutputCount;
            ++j)
        {
            printf("Answer: %f\n", TrainingData[i * TrainDataPitch + InputCount + j]);
            printf("Guess: %f\n\n", NeuralNetwork.Data[NeuralNetwork.DataRowPointer[LayerCount-1] + j]);
        }
        
        printf("\n");
	}
    
    printf("\n\n\n");
#endif

    u32 Correct = 0;
    u32 Incorrect = 0;

	for(u32 i=0; i<TEST_ITERATIONS; i++)
	{
		FeedForward(&NeuralNetwork,
                    &TrainingData[((i%TrainingDataPointCount)*TrainDataPitch)]);
        for(u32 j = 0;
            j < OutputCount;
            ++j)
        {
            u32 Actual = (u32)(TrainingData[(i%TrainingDataPointCount)*TrainDataPitch + InputCount + j]);
            r32 Guess = NeuralNetwork.Data[NeuralNetwork.DataRowPointer[LayerCount-1] + j];
            u32 Prediction = (u32)(Guess + 0.5f);
            if (Prediction == Actual)
            {
                ++Correct;
            } else {
                ++Incorrect;
            }
        }
	}

    r32 Accuracy = ((r32)Correct)/((r32)(Correct+Incorrect));

    printf("Correct: %d\n", Correct);
    printf("Incorrect: %d\n", Incorrect);
    printf("Accuracy: %.2f%%\n", Accuracy*100);
    
    return 1;
}
