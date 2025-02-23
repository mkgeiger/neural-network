/**********************************************************************************/
/* Copyright (c) 2025 Markus Geiger                                               */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "random.h"

/*---------------------------------------------------------------------------*/
/*     FUNCTION: random_init
**
**     brief    Seed the random number generator by the current time.
**
**     params   void
**     return   void
*/
/*---------------------------------------------------------------------------*/
void random_init(void)
{
    srand((unsigned int)time(NULL));
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: random_gaussian_real
**
**     brief    Generates a gaussian (normal) distributed real random number.
**
**              This function returns a gaussian distributed random number.
**              The Box Muller method is used.  
**
**     params   mean : mean (expected value)
**              stddev : standard deviation (sigma = sqrt(variance))  
**              min : minimum value
**              max : maximum value
**     return   gaussian random number
*/
/*---------------------------------------------------------------------------*/
real random_gaussian_real(real mean, real stddev, real min, real max)
{
    real u1, q, ret;
    static real u2;
    static int use_last = 0;

    do
    {
        if (use_last == 1)
        {
            u1 = u2;
            use_last = 0;
        }
        else
        {
            do
            {
                u1 = REAL_C(2.0) * (real)rand() / (real)RAND_MAX - REAL_C(1.0);
                u2 = REAL_C(2.0) * (real)rand() / (real)RAND_MAX - REAL_C(1.0);
                q = u1 * u1 + u2 * u2;
            }
            while ((q <= REAL_C(0.0)) || (q > REAL_C(1.0)));

            q = REAL_SQRT((-REAL_C(2.0) * REAL_LOG(q)) / q);
            u1 *= q;
            u2 *= q;
            use_last = 1;
        }

        ret = mean + u1 * stddev;
    }
    while ((ret < min) || (ret > max));

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: random_uniform_real
**
**     brief    Generates a uniform distributed real random number.
**
**              This function returns a uniform distributed random number.
**
**     params   min : minimum value
**              max : maximum value
**     return   uniform random number
*/
/*---------------------------------------------------------------------------*/
real random_uniform_real(real min, real max)
{
    return (min + (max - min) * (real)rand() / (real)RAND_MAX);
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: random_uniform_integer
**
**     brief    Generates a uniform distributed integer random number.
**
**              This function returns a uniform distributed random number.
**
**     params   min : minimum value
**              max : maximum value
**     return   uniform random number
*/
/*---------------------------------------------------------------------------*/
int16_t random_uniform_integer(int16_t min, int16_t max)
{
    return (min + (int16_t)(rand() % (max - min + 1)));
}
