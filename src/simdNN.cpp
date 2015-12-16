/* ========================================================================
   File: simdNN.cpp
   Date: 2015-12-15
   Creator: Steven Grissom
   ======================================================================== */

#define ALIGN_SIZE 16
#define ALIGN __declspec(align(ALIGN_SIZE))

__m128 One = _mm_set1_ps(1);
__m128 NegativeOne = _mm_set1_ps(-1);

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG r32 _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG s32 _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END

_PS_CONST(1  , 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS_CONST(cephes_log_p1, - 1.1514610310E-1f);
_PS_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS_CONST(cephes_log_p3, - 1.2420140846E-1f);
_PS_CONST(cephes_log_p4, + 1.4249322787E-1f);
_PS_CONST(cephes_log_p5, - 1.6668057665E-1f);
_PS_CONST(cephes_log_p6, + 2.0000714765E-1f);
_PS_CONST(cephes_log_p7, - 2.4999993993E-1f);
_PS_CONST(cephes_log_p8, + 3.3333331174E-1f);
_PS_CONST(cephes_log_q1, -2.12194440e-4f);
_PS_CONST(cephes_log_q2, 0.693359375f);

_PS_CONST(exp_hi,	88.3762626647949f);
_PS_CONST(exp_lo,	-88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

__m128 exp_ps(__m128 x) {
  __m128 tmp = _mm_setzero_ps(), fx;
  __m128i emm0;
  __m128 one = *(__m128*)_ps_1;

  x = _mm_min_ps(x, *(__m128*)_ps_exp_hi);
  x = _mm_max_ps(x, *(__m128*)_ps_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm_mul_ps(x, *(__m128*)_ps_cephes_LOG2EF);
  fx = _mm_add_ps(fx, *(__m128*)_ps_0p5);

  emm0 = _mm_cvttps_epi32(fx);
  tmp  = _mm_cvtepi32_ps(emm0);
  /* if greater, substract 1 */
  __m128 mask = _mm_cmpgt_ps(tmp, fx);    
  mask = _mm_and_ps(mask, one);
  fx = _mm_sub_ps(tmp, mask);

  tmp = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C1);
  __m128 z = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C2);
  x = _mm_sub_ps(x, tmp);
  x = _mm_sub_ps(x, z);

  z = _mm_mul_ps(x,x);
  
  __m128 y = *(__m128*)_ps_cephes_exp_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p5);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, x);
  y = _mm_add_ps(y, one);

  emm0 = _mm_cvttps_epi32(fx);
  emm0 = _mm_add_epi32(emm0, *(__m128i*)_pi32_0x7f);
  emm0 = _mm_slli_epi32(emm0, 23);
  __m128 pow2n = _mm_castsi128_ps(emm0);
  y = _mm_mul_ps(y, pow2n);
  return y;
}

void PrintWide(__m128 WideFloat)
{
    r32 *Floats = (r32*) &WideFloat;
    printf("%.3f %.3f %.3f %.3f\n", 
           Floats[0], Floats[1], Floats[2], Floats[3]);
}

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
            r32 *Result = (r32 *)AlignedMalloc(sizeof(r32) * 16);
            
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
                
                r32 c0 = 0.0f;
                r32 c1 = 0.0f;
                r32 c2 = 0.0f;
                r32 c3 = 0.0f;

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

                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        InvalidCodePath;
                        
                        c0 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+0) +
                                     PrevNeuronIndex]);
                        c1 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+1) +
                                     PrevNeuronIndex]);
                        c2 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+2) +
                                     PrevNeuronIndex]);
                        c3 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+3) +
                                     PrevNeuronIndex]);
                    }
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
                                                  _mm_add_ps(One, exp_ps(
                                                                 _mm_mul_ps(NegativeOne, WideSumTotal))));

                

                _mm_store_ps(Data+DataRowPtr[LayerIndex]+(NeuronIndex), SigmoidResult);
                
                /*
                Sum0 += c00 +
                    Result[0] + Result[1] + Result[2] + Result[3] +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+0) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+0)] = (1.0f/(1.0f+(r32)exp(-Sum0)));
                
                Sum1 += c10 +
                    Result[4] + Result[5] + Result[6] + Result[7] +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+1) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+1)] = (1.0f/(1.0f+(r32)exp(-Sum1)));
                
                Sum2 += c20 +
                    Result[8] + Result[9] + Result[10] + Result[11] +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+2) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+2)] = (1.0f/(1.0f+(r32)exp(-Sum2)));
                
                Sum3 += c30 +
                    Result[12] + Result[13] + Result[14] + Result[15] +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+3) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+3)] = (1.0f/(1.0f+(r32)exp(-Sum3)));
                */
            }
            
            free(Result);
            
            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                r32 Sum = 0.0f;
                r32 c0 = 0.0f;
                r32 c1 = 0.0f;
                r32 c2 = 0.0f;
                r32 c3 = 0.0f;
            
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        c0 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex]);
                        c1 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+1]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex+1]);
                        c2 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+2]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex+2]);
                        c3 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+3]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex+3]);
                    }
            
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        c0 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex]);
                    }
                }
            
                Sum += c0 + c1 + c2 + c3 +
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
    FeedForward(NeuralNetwork, DataPoint);
    
    r32 *Data = NeuralNetwork->Data;
    r32 *Delta = NeuralNetwork->Delta;
    r32 *Weights = NeuralNetwork->Weights;
    r32 *WeightDelta = NeuralNetwork->WeightDelta;
    u32 *DataRowPtr = NeuralNetwork->DataRowPointer;
    u32 *WeightsRowPtr = NeuralNetwork->WeightsRowPointer;
    u32 *LayerSizes = NeuralNetwork->LayerSizes;
    u32 LayerCount = NeuralNetwork->LayerCount;
    r32 Alpha = NeuralNetwork->Alpha;
    r32 Beta = NeuralNetwork->Beta;

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
                r32 c00 = 0.0f;
                r32 c01 = 0.0f;
                r32 c02 = 0.0f;
                r32 c03 = 0.0f;

                r32 Sum1 = 0.0f;
                r32 c10 = 0.0f;
                r32 c11 = 0.0f;
                r32 c12 = 0.0f;
                r32 c13 = 0.0f;

                r32 Sum2 = 0.0f;
                r32 c20 = 0.0f;
                r32 c21 = 0.0f;
                r32 c22 = 0.0f;
                r32 c23 = 0.0f;

                r32 Sum3 = 0.0f;
                r32 c30 = 0.0f;
                r32 c31 = 0.0f;
                r32 c32 = 0.0f;
                r32 c33 = 0.0f;

                {
                    u32 NextNeuronIndex=0;
                    for(;
                        NextNeuronIndex+3 < LayerSizes[LayerIndex+1];
                        NextNeuronIndex+=4)
                    {
                        c00 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+0)];
                        c01 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+0)];
                        c02 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+0)];
                        c03 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+0)];
                    
                        c10 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+1)];
                        c11 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+1)];
                        c12 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+1)];
                        c13 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+1)];
                    
                        c20 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+2)];
                        c21 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+2)];
                        c22 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+2)];
                        c23 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+2)];
                    
                        c30 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+3)];
                        c31 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+3)];
                        c32 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+3)];
                        c33 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+3)];
                    }
                    for(;
                        NextNeuronIndex < LayerSizes[LayerIndex+1];
                        ++NextNeuronIndex)
                    {
                        c00 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+0)];
                        c10 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+1)];
                        c20 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+2)];
                        c30 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+3)];
                    }
                }

                Sum0 += c00 + c01 + c02 + c03;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+0)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+0)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+0)]) *
                    Sum0;
                
                Sum1 += c10 + c11 + c12 + c13;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+1)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+1)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+1)]) *
                    Sum1;
                
                Sum2 += c20 + c21 + c22 + c23;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+2)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+2)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+2)]) *
                    Sum2;
                
                Sum3 += c30 + c31 + c32 + c33;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+3)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+3)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+3)]) *
                    Sum3;
            }
            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                r32 Sum = 0.0f;
                r32 c0 = 0.0f;
                r32 c1 = 0.0f;
                r32 c2 = 0.0f;
                r32 c3 = 0.0f;

                {
                    u32 NextNeuronIndex=0;
                    for(;
                        NextNeuronIndex+3 < LayerSizes[LayerIndex+1];
                        NextNeuronIndex+=4)
                    {
                        c0 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    NeuronIndex];
                        c1 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    NeuronIndex];
                        c2 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    NeuronIndex];
                        c3 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    NeuronIndex];
                    }
                    for(;
                        NextNeuronIndex < LayerSizes[LayerIndex+1];
                        ++NextNeuronIndex)
                    {
                        c0 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    NeuronIndex];
                    }
                }

                Sum += c0 + c1 + c2 + c3;

                Delta[DataRowPtr[LayerIndex] + NeuronIndex] =
                    Data[DataRowPtr[LayerIndex] + NeuronIndex] *
                    (1 - Data[DataRowPtr[LayerIndex] + NeuronIndex]) *
                    Sum;
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
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 3];
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 3];
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 3];
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 3];
                    }
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex];
                    
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex];
                    
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex];
                    
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex];
                    }
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
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+3)];


                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+3)];



                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+3)];


                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+3)];

                    }
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + PrevNeuronIndex];
                        
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex];
                        
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex];
                        
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex];
                        
                    }
                }

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+0))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+0)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+0)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * (NeuronIndex+0) + LayerSizes[LayerIndex-1]];

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+1))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+1)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * (NeuronIndex+1) + LayerSizes[LayerIndex-1]];

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+2))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+2)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * (NeuronIndex+2) + LayerSizes[LayerIndex-1]];

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+3))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+3)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
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
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+3)];
                    }
                    
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + PrevNeuronIndex];
                    }
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
