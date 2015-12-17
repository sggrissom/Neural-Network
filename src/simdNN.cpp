/* ========================================================================
   File: simdNN.cpp
   Date: 2015-12-15
   Creator: Steven Grissom
   ======================================================================== */

#include "simd.h"

internal void
FeedForward(neural_network *NeuralNetwork, r32 *DataPoint)
{
    r32 *Data = NeuralNetwork->Data;
    r32 *Weights = NeuralNetwork->Weights;
    u32 *DataRowPtr = NeuralNetwork->DataRowPointer;
    u32 *WeightsRowPtr = NeuralNetwork->WeightsRowPointer;
    u32 *LayerSizes = NeuralNetwork->LayerSizes;

    {
        u32 InputIndex = 0;
        for (;
             InputIndex+3 < NeuralNetwork->InputCount;
             InputIndex += 4)
        {
            Data[DataRowPtr[0]+InputIndex] = DataPoint[InputIndex];
            Data[DataRowPtr[0]+InputIndex+1] = DataPoint[InputIndex+1];
            Data[DataRowPtr[0]+InputIndex+2] = DataPoint[InputIndex+2];
            Data[DataRowPtr[0]+InputIndex+3] = DataPoint[InputIndex+3];
        }
    
        for (;
             InputIndex < NeuralNetwork->InputCount;
             ++InputIndex)
        {
            Data[DataRowPtr[0]+InputIndex] = DataPoint[InputIndex];
        }
    }

    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
	{
        u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
        while(PrevLayerSize&15)
        {
            ++PrevLayerSize;
        }
     
        {
            u32 NeuronIndex = 0;
            
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                r32 Sum0 = 0.0f;
                __m128 WideSum0 = _mm_set1_ps(0.0f);
                r32 Sum1 = 0.0f;             
                __m128 WideSum1 = _mm_set1_ps(0.0f);
                r32 Sum2 = 0.0f;             
                __m128 WideSum2 = _mm_set1_ps(0.0f);
                r32 Sum3 = 0.0f;
                __m128 WideSum3 = _mm_set1_ps(0.0f);

                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        __m128 Value = _mm_load_ps(Data + DataRowPtr[LayerIndex-1]+PrevNeuronIndex);

                        __m128 Weight0 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+0) +
                                                     PrevNeuronIndex);
                        WideSum0 = _mm_add_ps(WideSum0, _mm_mul_ps(Value, Weight0));

                        __m128 Weight1 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+1) +
                                                     PrevNeuronIndex);
                        WideSum1 = _mm_add_ps(WideSum1, _mm_mul_ps(Value, Weight1));
                        
                        __m128 Weight2 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+2) +
                                                     PrevNeuronIndex);
                        WideSum2 = _mm_add_ps(WideSum2, _mm_mul_ps(Value, Weight2));
                        
                        __m128 Weight3 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+3) +
                                                     PrevNeuronIndex);
                        WideSum3 = _mm_add_ps(WideSum3, _mm_mul_ps(Value, Weight3));
                    }
                    
                    Assert(PrevNeuronIndex >= LayerSizes[LayerIndex-1]);
                }

                __m128 WideSum01 = _mm_hadd_ps(WideSum1, WideSum0);
                __m128 WideSum23 = _mm_hadd_ps(WideSum3, WideSum2);

                __m128 WideSumTotal = _mm_hadd_ps(WideSum23, WideSum01);

                WideSumTotal = _mm_shuffle_ps(WideSumTotal, WideSumTotal, _MM_SHUFFLE(0,1,2,3));
                
                __m128 Leftovers = _mm_set_ps(
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+3) + LayerSizes[LayerIndex-1]],
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+2) + LayerSizes[LayerIndex-1]],
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+1) + LayerSizes[LayerIndex-1]],
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+0) + LayerSizes[LayerIndex-1]]);

                WideSumTotal = _mm_add_ps(WideSumTotal, Leftovers);
                
                __m128 SigmoidResult = _mm_div_ps(One,
                                                  _mm_add_ps(One,
                                                             exp_ps(
                                                                 _mm_mul_ps(NegativeOne, WideSumTotal))));

                
                _mm_store_ps(Data+DataRowPtr[LayerIndex]+(NeuronIndex), SigmoidResult);
            }
            

            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                __m128 WideSum = _mm_set1_ps(0.0f);
            
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        __m128 Value = _mm_load_ps(Data + DataRowPtr[LayerIndex-1]+PrevNeuronIndex);

                        __m128 Weight = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                    (PrevLayerSize)*(NeuronIndex) +
                                                    PrevNeuronIndex);
                        WideSum = _mm_add_ps(WideSum, _mm_mul_ps(Value, Weight));
                    }

                    Assert(PrevNeuronIndex >= LayerSizes[LayerIndex-1]);
                }

                ALIGN r32 Result;

                WideSum = _mm_hadd_ps(WideSum, WideSum);
                WideSum = _mm_hadd_ps(WideSum, WideSum);

                _mm_store_ps(&Result, WideSum);

                r32 Sum = Result +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*NeuronIndex +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+NeuronIndex] = (1.0f/(1.0f+(r32)exp(-Sum)));
            }
        }
	}
}

internal void
BackPropogate(neural_network *NeuralNetwork, r32 *DataPoint, r32 *Target)
{

    
    r32 *Data = NeuralNetwork->Data;
    r32 *Weights = NeuralNetwork->Weights;
    u32 *DataRowPtr = NeuralNetwork->DataRowPointer;
    u32 *WeightsRowPtr = NeuralNetwork->WeightsRowPointer;
    u32 *LayerSizes = NeuralNetwork->LayerSizes;
    r32 *Delta = NeuralNetwork->Delta;
    r32 *WeightDelta = NeuralNetwork->WeightDelta;
    u32 LayerCount = NeuralNetwork->LayerCount;
    r32 Alpha = NeuralNetwork->Alpha;
    r32 Beta = NeuralNetwork->Beta;

    __m128 WideAlpha = _mm_set1_ps(Alpha);
    __m128 WideBeta = _mm_set1_ps(Beta);
        
/****************************************************************/
//Inline Feed Forward    

    {
        u32 InputIndex = 0;
        for (;
             InputIndex+3 < NeuralNetwork->InputCount;
             InputIndex += 4)
        {
            Data[DataRowPtr[0]+InputIndex] = DataPoint[InputIndex];
            Data[DataRowPtr[0]+InputIndex+1] = DataPoint[InputIndex+1];
            Data[DataRowPtr[0]+InputIndex+2] = DataPoint[InputIndex+2];
            Data[DataRowPtr[0]+InputIndex+3] = DataPoint[InputIndex+3];
        }
    
        for (;
             InputIndex < NeuralNetwork->InputCount;
             ++InputIndex)
        {
            Data[DataRowPtr[0]+InputIndex] = DataPoint[InputIndex];
        }
    }

    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
	{
        u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
        while(PrevLayerSize&15)
        {
            ++PrevLayerSize;
        }
     
        {
            u32 NeuronIndex = 0;
            
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                r32 Sum0 = 0.0f;
                __m128 WideSum0 = _mm_set1_ps(0.0f);
                r32 Sum1 = 0.0f;             
                __m128 WideSum1 = _mm_set1_ps(0.0f);
                r32 Sum2 = 0.0f;             
                __m128 WideSum2 = _mm_set1_ps(0.0f);
                r32 Sum3 = 0.0f;
                __m128 WideSum3 = _mm_set1_ps(0.0f);

                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        __m128 Value = _mm_load_ps(Data + DataRowPtr[LayerIndex-1]+PrevNeuronIndex);

                        __m128 Weight0 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+0) +
                                                     PrevNeuronIndex);
                        WideSum0 = _mm_add_ps(WideSum0, _mm_mul_ps(Value, Weight0));

                        __m128 Weight1 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+1) +
                                                     PrevNeuronIndex);
                        WideSum1 = _mm_add_ps(WideSum1, _mm_mul_ps(Value, Weight1));
                        
                        __m128 Weight2 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+2) +
                                                     PrevNeuronIndex);
                        WideSum2 = _mm_add_ps(WideSum2, _mm_mul_ps(Value, Weight2));
                        
                        __m128 Weight3 = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                     (PrevLayerSize)*(NeuronIndex+3) +
                                                     PrevNeuronIndex);
                        WideSum3 = _mm_add_ps(WideSum3, _mm_mul_ps(Value, Weight3));
                    }
                    
                    Assert(PrevNeuronIndex >= LayerSizes[LayerIndex-1]);
                }

                __m128 WideSum01 = _mm_hadd_ps(WideSum1, WideSum0);
                __m128 WideSum23 = _mm_hadd_ps(WideSum3, WideSum2);

                __m128 WideSumTotal = _mm_hadd_ps(WideSum23, WideSum01);

                WideSumTotal = _mm_shuffle_ps(WideSumTotal, WideSumTotal, _MM_SHUFFLE(0,1,2,3));
                
                __m128 Leftovers = _mm_set_ps(
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+3) + LayerSizes[LayerIndex-1]],
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+2) + LayerSizes[LayerIndex-1]],
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+1) + LayerSizes[LayerIndex-1]],
                    Weights[WeightsRowPtr[LayerIndex] + (PrevLayerSize)*(NeuronIndex+0) + LayerSizes[LayerIndex-1]]);

                WideSumTotal = _mm_add_ps(WideSumTotal, Leftovers);
                
                __m128 SigmoidResult = _mm_div_ps(One,
                                                  _mm_add_ps(One,
                                                             exp_ps(
                                                                 _mm_mul_ps(NegativeOne, WideSumTotal))));

                
                _mm_store_ps(Data+DataRowPtr[LayerIndex]+(NeuronIndex), SigmoidResult);
            }
            

            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                __m128 WideSum = _mm_set1_ps(0.0f);
            
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        __m128 Value = _mm_load_ps(Data + DataRowPtr[LayerIndex-1]+PrevNeuronIndex);

                        __m128 Weight = _mm_load_ps(Weights + WeightsRowPtr[LayerIndex] +
                                                    (PrevLayerSize)*(NeuronIndex) +
                                                    PrevNeuronIndex);
                        WideSum = _mm_add_ps(WideSum, _mm_mul_ps(Value, Weight));
                    }

                    Assert(PrevNeuronIndex >= LayerSizes[LayerIndex-1]);
                }

                ALIGN r32 Result;

                WideSum = _mm_hadd_ps(WideSum, WideSum);
                WideSum = _mm_hadd_ps(WideSum, WideSum);

                _mm_store_ps(&Result, WideSum);

                r32 Sum = Result +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*NeuronIndex +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+NeuronIndex] = (1.0f/(1.0f+(r32)exp(-Sum)));
            }
        }
	}
    
/****************************************************************/
    
    {
        u32 LayerIndex = 0;
        for (;
             LayerIndex+3 < LayerSizes[(LayerCount)-1];
             LayerIndex+=4)
        {
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]) *
                (Target[LayerIndex] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]);
            
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex+1] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex+1] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+1]) *
                (Target[LayerIndex+1] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+1]);
            
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex+2] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex+2] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+2]) *
                (Target[LayerIndex+2] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+2]);
            
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex+3] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex+3] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+3]) *
                (Target[LayerIndex+3] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+3]);
        }
        for (;
             LayerIndex < LayerSizes[(LayerCount)-1];
             ++LayerIndex)
        {
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]) *
                (Target[LayerIndex] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]);
        }
    }

/****************************************************************/
    
    for(u32 LayerIndex = LayerCount-2;
        LayerIndex > 0;
        --LayerIndex)
    {
        u32 CurrLayerSize = LayerSizes[LayerIndex]+1;
        while(CurrLayerSize&15)
        {
            ++CurrLayerSize;
        }
        {
            u32 NeuronIndex = 0;
                        
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                r32 Sum0 = 0.0f;
                __m128 WideSum0 = _mm_set1_ps(0.0f);
                r32 Sum1 = 0.0f;             
                __m128 WideSum1 = _mm_set1_ps(0.0f);
                r32 Sum2 = 0.0f;             
                __m128 WideSum2 = _mm_set1_ps(0.0f);
                r32 Sum3 = 0.0f;
                __m128 WideSum3 = _mm_set1_ps(0.0f);
                
                __m128 RemainingSum = _mm_set1_ps(0.0f);
                
                {
                    u32 NextNeuronIndex=0;
                    for(;
                        NextNeuronIndex+3 < LayerSizes[LayerIndex+1];
                        NextNeuronIndex+=4)
                    {
                        __m128 Value0 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex+1] +
                                                          NextNeuronIndex]);

                        __m128 Value1 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex+1] +
                                                          NextNeuronIndex+1]);

                        __m128 Value2 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex+1] +
                                                          NextNeuronIndex+2]);

                        __m128 Value3 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex+1] +
                                                          NextNeuronIndex+3]);

                        __m128 Weight0 = _mm_load_ps(Weights +
                                                     WeightsRowPtr[LayerIndex+1] +
                                                     (CurrLayerSize)*NextNeuronIndex +
                                                     NeuronIndex);
                        WideSum0 = _mm_add_ps(WideSum0, _mm_mul_ps(Value0, Weight0));
                        
                        __m128 Weight1 = _mm_load_ps(Weights +
                                                     WeightsRowPtr[LayerIndex+1] +
                                                     (CurrLayerSize)*(NextNeuronIndex+1) +
                                                     NeuronIndex);
                        WideSum1 = _mm_add_ps(WideSum1, _mm_mul_ps(Value1, Weight1));
                        
                        __m128 Weight2 = _mm_load_ps(Weights +
                                                     WeightsRowPtr[LayerIndex+1] +
                                                     (CurrLayerSize)*(NextNeuronIndex+2) +
                                                     NeuronIndex);
                        WideSum2 = _mm_add_ps(WideSum2, _mm_mul_ps(Value2, Weight2));
                        
                        __m128 Weight3 = _mm_load_ps(Weights +
                                                     WeightsRowPtr[LayerIndex+1] +
                                                     (CurrLayerSize)*(NextNeuronIndex+3) +
                                                     NeuronIndex);
                        WideSum3 = _mm_add_ps(WideSum3, _mm_mul_ps(Value3, Weight3));

                    }
                    
                    for(;
                        NextNeuronIndex < LayerSizes[LayerIndex+1];
                        ++NextNeuronIndex)
                    {
                        __m128 WideDelta = _mm_set1_ps(Delta[DataRowPtr[LayerIndex+1]+
                                                             NextNeuronIndex]);

                        __m128 WideWeight = _mm_load_ps(Weights +
                                                        WeightsRowPtr[LayerIndex+1] +
                                                        (CurrLayerSize)*NextNeuronIndex +
                                                        (NeuronIndex));

                        RemainingSum = _mm_add_ps(RemainingSum, _mm_mul_ps(WideDelta, WideWeight));
                    }
                }
                
                __m128 WideSumTotal = _mm_add_ps(WideSum0, _mm_add_ps(WideSum1, _mm_add_ps(WideSum2, WideSum3)));

                WideSumTotal = _mm_add_ps(WideSumTotal, RemainingSum);

                __m128 WideData = _mm_load_ps(Data + DataRowPtr[LayerIndex] + NeuronIndex);
                __m128 InvWideData = _mm_sub_ps(One, WideData);
                __m128 StoreValue = _mm_mul_ps(WideData, _mm_mul_ps(InvWideData, WideSumTotal));

                _mm_store_ps(Delta + DataRowPtr[LayerIndex] + NeuronIndex,
                             StoreValue);
            }

            Assert(NeuronIndex >= LayerSizes[LayerIndex]);
        }
    }

/****************************************************************/

	for(u32 LayerIndex = 1;
        LayerIndex < LayerCount;
        ++LayerIndex)
	{
        u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
        while(PrevLayerSize&15)
        {
            ++PrevLayerSize;
        }

        {
            u32 NeuronIndex = 0;
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                {
                    u32 PrevNeuronIndex = 0;

                    
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        __m128 WideWeightDx0 = _mm_load_ps(WeightDelta +
                                                           WeightsRowPtr[LayerIndex] +
                                                           PrevLayerSize * (NeuronIndex+0) +
                                                           PrevNeuronIndex);
                        __m128 WideWeightDx1 = _mm_load_ps(WeightDelta +
                                                           WeightsRowPtr[LayerIndex] +
                                                           PrevLayerSize * (NeuronIndex+1) +
                                                           PrevNeuronIndex);
                        __m128 WideWeightDx2 = _mm_load_ps(WeightDelta +
                                                           WeightsRowPtr[LayerIndex] +
                                                           PrevLayerSize * (NeuronIndex+2) +
                                                           PrevNeuronIndex);
                        __m128 WideWeightDx3 = _mm_load_ps(WeightDelta +
                                                           WeightsRowPtr[LayerIndex] +
                                                           PrevLayerSize * (NeuronIndex+3) +
                                                           PrevNeuronIndex);

                        __m128 WideWeight0 = _mm_load_ps(Weights +
                                                         WeightsRowPtr[LayerIndex] +
                                                         PrevLayerSize * (NeuronIndex+0) +
                                                         PrevNeuronIndex);
                        __m128 WideWeight1 = _mm_load_ps(Weights +
                                                         WeightsRowPtr[LayerIndex] +
                                                         PrevLayerSize * (NeuronIndex+1) +
                                                         PrevNeuronIndex);
                        __m128 WideWeight2 = _mm_load_ps(Weights +
                                                         WeightsRowPtr[LayerIndex] +
                                                         PrevLayerSize * (NeuronIndex+2) +
                                                         PrevNeuronIndex);
                        __m128 WideWeight3 = _mm_load_ps(Weights +
                                                         WeightsRowPtr[LayerIndex] +
                                                         PrevLayerSize * (NeuronIndex+3) +
                                                         PrevNeuronIndex);

                        WideWeight0 = _mm_add_ps(WideWeight0, _mm_mul_ps(WideWeightDx0, WideAlpha));
                        WideWeight1 = _mm_add_ps(WideWeight1, _mm_mul_ps(WideWeightDx1, WideAlpha));
                        WideWeight2 = _mm_add_ps(WideWeight2, _mm_mul_ps(WideWeightDx2, WideAlpha));
                        WideWeight3 = _mm_add_ps(WideWeight3, _mm_mul_ps(WideWeightDx3, WideAlpha));

                        _mm_store_ps(Weights +
                                     WeightsRowPtr[LayerIndex] +
                                     PrevLayerSize * (NeuronIndex+0) +
                                     PrevNeuronIndex,
                                     WideWeight0);
                        _mm_store_ps(Weights +
                                     WeightsRowPtr[LayerIndex] +
                                     PrevLayerSize * (NeuronIndex+1) +
                                     PrevNeuronIndex,
                                     WideWeight1);
                        _mm_store_ps(Weights +
                                     WeightsRowPtr[LayerIndex] +
                                     PrevLayerSize * (NeuronIndex+2) +
                                     PrevNeuronIndex,
                                     WideWeight2);
                        _mm_store_ps(Weights +
                                     WeightsRowPtr[LayerIndex] +
                                     PrevLayerSize * (NeuronIndex+3) +
                                     PrevNeuronIndex,
                                     WideWeight3);
                    }
                    
                    Assert(PrevNeuronIndex >= LayerSizes[LayerIndex-1]);
                }

                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+0) + LayerSizes[LayerIndex-1]];
                
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + LayerSizes[LayerIndex-1]];
                
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + LayerSizes[LayerIndex-1]];
                
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + LayerSizes[LayerIndex-1]];
            }
        
            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex + 3];
                    }
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex];
                    }
                }

                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]];
            }
        }
	}

/****************************************************************/
    
    
	for(u32 LayerIndex = 1;
        LayerIndex < LayerCount;
        ++LayerIndex)
	{
        u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
        while(PrevLayerSize&15)
        {
            ++PrevLayerSize;
        }

        {
            u32 NeuronIndex = 0;
            
            __m128 WideWeight0 = _mm_set1_ps(0.0f);
            __m128 WideWeight1 = _mm_set1_ps(0.0f);
            __m128 WideWeight2 = _mm_set1_ps(0.0f);
            __m128 WideWeight3 = _mm_set1_ps(0.0f);
            __m128 WideBetaWeight = _mm_set1_ps(0.0f);
        
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        __m128 WideDelta0 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex] +
                                                       NeuronIndex]);
                        __m128 WideData = _mm_load_ps(Data +
                                                      DataRowPtr[LayerIndex-1] +
                                                      (PrevNeuronIndex));

                        __m128 WideWeightDx0 = _mm_mul_ps(WideBeta, _mm_mul_ps(WideData, WideDelta0));

                        _mm_store_ps(WeightDelta + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * NeuronIndex) +
                                     PrevNeuronIndex, WideWeightDx0);

                        

                        __m128 WideDelta1 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex] +
                                                        NeuronIndex+1]);

                        __m128 WideWeightDx1 = _mm_mul_ps(WideBeta, _mm_mul_ps(WideData, WideDelta1));

                        _mm_store_ps(WeightDelta + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex+1)) +
                                     PrevNeuronIndex, WideWeightDx1);

                        
                        __m128 WideDelta2 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex] +
                                                        NeuronIndex+2]);

                        __m128 WideWeightDx2 = _mm_mul_ps(WideBeta, _mm_mul_ps(WideData, WideDelta2));

                        _mm_store_ps(WeightDelta + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex+2)) +
                                     PrevNeuronIndex, WideWeightDx2);

                        
                        __m128 WideDelta3 = _mm_set1_ps(Delta[DataRowPtr[LayerIndex] +
                                                        NeuronIndex+3]);

                        __m128 WideWeightDx3 = _mm_mul_ps(WideBeta, _mm_mul_ps(WideData, WideDelta3));

                        _mm_store_ps(WeightDelta + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex+3)) +
                                     PrevNeuronIndex, WideWeightDx3);

                        WideWeight0 = _mm_add_ps(WideWeight0, WideWeightDx0);
                        WideWeight1 = _mm_add_ps(WideWeight1, WideWeightDx1);
                        WideWeight2 = _mm_add_ps(WideWeight2, WideWeightDx2);
                        WideWeight3 = _mm_add_ps(WideWeight3, WideWeightDx3);
                        
/*                        
                        _mm_store_ps(Weights + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex+0)) +
                                     PrevNeuronIndex,
                                     WideWeight0);
                        _mm_store_ps(Weights + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex+1)) +
                                     PrevNeuronIndex,
                                     WideWeight1);
                        _mm_store_ps(Weights + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex+2)) +
                                     PrevNeuronIndex,
                                     WideWeight2);
                        _mm_store_ps(Weights + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex+3)) +
                                     PrevNeuronIndex,
                                     WideWeight3);
*/
                    }

                    Assert(PrevNeuronIndex >= LayerSizes[LayerIndex-1]);
                }

                __m128 WideDelta = _mm_load_ps(Delta +
                                               DataRowPtr[LayerIndex] +
                                               NeuronIndex);
            
                WideDelta = _mm_mul_ps(WideDelta, WideBeta);

                ALIGN r32 Result[4];

                _mm_store_ps(Result, WideDelta);
                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+0))
                            + LayerSizes[LayerIndex-1]] = Result[0];
                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+1))
                            + LayerSizes[LayerIndex-1]] = Result[1];
            
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+2))
                            + LayerSizes[LayerIndex-1]] = Result[2];
                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+3))
                            + LayerSizes[LayerIndex-1]] = Result[3];

                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+0)) +
                        LayerSizes[LayerIndex-1]] += Result[0];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+1)) +
                        LayerSizes[LayerIndex-1]] += Result[1];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+2)) +
                        LayerSizes[LayerIndex-1]] += Result[2];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+3)) +
                        LayerSizes[LayerIndex-1]] += Result[3];
            }

            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {

                __m128 WideWeight = _mm_set1_ps(0.0f);
                
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        __m128 WideDelta = _mm_set1_ps(Delta[DataRowPtr[LayerIndex]+NeuronIndex]);
                        __m128 WideData = _mm_load_ps(Data + DataRowPtr[LayerIndex-1] + PrevNeuronIndex);

                        __m128 WideWeightDx = _mm_mul_ps(WideBeta, _mm_mul_ps(WideData, WideDelta));

                        _mm_store_ps(WeightDelta + WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                     + PrevNeuronIndex,
                                     WideWeightDx);

                        WideWeight = _mm_add_ps(WideWeight, WideWeightDx);

                        _mm_store_ps(Weights + WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize * (NeuronIndex)) +
                                     PrevNeuronIndex,
                                     WideWeight);
                    }

                    Assert(PrevNeuronIndex >= LayerSizes[LayerIndex-1]);
                }
                
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * NeuronIndex)
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * NeuronIndex) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]];
            }
        }
	}
}
