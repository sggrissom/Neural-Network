
#include "../../slib/slib.h"

#define INPUTCOUNT 2
#define OUTPUTCOUNT 1
#define LAYERSIZES {2,3,1}
#define BETA 0.3f
#define ALPHA 0.1f
#define EPSILON 0.0001f
#define MAXITERATIONS 500000

    global r32 TrainingData[] = {
        0,0,0,
        0,1,1,
        1,0,1,
        1,1,0,
    };
    
#include "FlatNN.h"
//#include "SimpleNN.h"
//#include "transformNN.h"

s32 main()
{
    FlatNN();
//    SimpleNN();
//    TransformNN();

    return 1;
}
