/* ========================================================================
   File: load.cpp
   Date: 2015-12-14
   Creator: Steven Grissom
   ======================================================================== */

internal u32
LoadCSV(char *Filename, r32 **ArrayRef)
{
    FILE *File;
    fopen_s(&File, Filename, "r");

    u32 ArraySize = 10;
    r32 *Values = (r32 *)malloc(sizeof(r32) * 10);

    AssertAligned(Values);

    u32 ArrayUsed = 0;
    
    if(File)
    {
        s32 Character = fgetc(File);
        char FloatString[6];

        while(Character != EOF)
        {
            u32 i = 0;
            char *String = FloatString;
            while(Character != ',' &&
                  Character != '\n' &&
                  Character != EOF)
            {
                Assert(i++ < 6);
                *String++ = (char)Character;
                Character = fgetc(File);
            }

            *String = 0;

            r32 FloatValue = (r32)atof((char *)FloatString);

            Values[ArrayUsed++] = FloatValue;

            if(ArrayUsed == ArraySize)
            {
                ArraySize *= 2;
                Values = (r32 *)realloc(Values, sizeof(r32) * ArraySize);

                AssertAligned(Values);
            }
            
            Character = fgetc(File);
        }

        fclose(File);
    }

    *ArrayRef = Values;

    return ArrayUsed;
}
