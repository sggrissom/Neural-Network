
#include <time.h>
#include <math.h>
#include <omp.h>

#include <slib.h>

#include "NeuralNetwork.cpp"

#if FLAT
#include "FlatNN.cpp"
#elif UNROLLED
#include "UnrolledNN.cpp"
#elif SIMD
#include "simdNN.cpp"
#endif

internal void
CombineArrays(r32 *A, r32 *B, r32 *C, r32 *D, u32 Size)
{
    for(u32 ArrayIndex = 0;
        ArrayIndex < Size;
        ++ArrayIndex)
    {
        A[ArrayIndex] += B[ArrayIndex] + C[ArrayIndex] + D[ArrayIndex];
        A[ArrayIndex] /= 4;
    }
}

s32 main()
{
    srand(0);
    
    neural_network NeuralNetwork[NETWORK_COUNT] = {};
    
    u32 LayerSizes[] = LAYERSIZES;
    u32 MaximumIterations =  MAX_ITERATIONS / NETWORK_COUNT;
    u32 LayerCount = ArrayCount(LayerSizes);
    u32 InputCount = LayerSizes[0];
    u32 OutputCount = LayerSizes[LayerCount - 1];

    u32 WeightCount = 0;
    
    for(u32 NetworkCount = 0;
        NetworkCount < NETWORK_COUNT;
        ++NetworkCount)
    {
        NeuralNetwork[NetworkCount].LayerSizes = LayerSizes;
        NeuralNetwork[NetworkCount].LayerCount = LayerCount;
        NeuralNetwork[NetworkCount].InputCount = InputCount;
        NeuralNetwork[NetworkCount].OutputCount = OutputCount;
        NeuralNetwork[NetworkCount].Beta = BETA;
        NeuralNetwork[NetworkCount].Alpha = ALPHA;
        NeuralNetwork[NetworkCount].Epsilon = EPSILON;

        WeightCount = InitializeNetwork(&NeuralNetwork[NetworkCount]);
    }

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

    clock_t startTime = clock();

#pragma omp parallel num_threads(NETWORK_COUNT)
    {
#if MP
        u32 NetworkIndex = omp_get_thread_num();
#else
        u32 NetworkIndex = 0;
#endif

        for(u32 IterationIndex = 0;
            IterationIndex < MaximumIterations*TrainDataPitch;
            ++IterationIndex)
        {
            r32 *DataPoint = &TrainingData[(IterationIndex%TrainingDataPointCount)*TrainDataPitch];
            r32 *Target = DataPoint + InputCount;
            BackPropogate(&NeuralNetwork[NetworkIndex], DataPoint, Target);
        }
    }

#if MP
    u32 NeuronCount = NeuralNetwork[0].DataRowPointer[LayerCount];
    CombineArrays(NeuralNetwork[0].Data,
                  NeuralNetwork[1].Data,
                  NeuralNetwork[2].Data,
                  NeuralNetwork[3].Data,
                  NeuronCount);
    CombineArrays(NeuralNetwork[0].Weights,
                  NeuralNetwork[1].Weights,
                  NeuralNetwork[2].Weights,
                  NeuralNetwork[3].Weights,
                  WeightCount);
#endif

    clock_t endTime = clock();
    clock_t clockTicksTaken = endTime - startTime;
    r32 timeInSeconds = clockTicksTaken / (r32) CLOCKS_PER_SEC;
    
    printf("%.3f s\n", timeInSeconds);

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
		FeedForward(&NeuralNetwork[0],
                    &TrainingData[((i%TrainingDataPointCount)*TrainDataPitch)]);
        for(u32 j = 0;
            j < OutputCount;
            ++j)
        {
            u32 Actual = (u32)(TrainingData[(i%TrainingDataPointCount)*TrainDataPitch + InputCount + j]);
            r32 Guess = NeuralNetwork[0].Data[NeuralNetwork[0].DataRowPointer[LayerCount-1] + j];
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
