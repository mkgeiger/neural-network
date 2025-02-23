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

#include <stdint.h>
#include <stdlib.h>

#include "alloc.h"

size_t bytes_allocated = 0;

/*---------------------------------------------------------------------------*/
/*     FUNCTION: alloc_aligned_malloc
**
**     brief    Allocates number of bytes with alignment.
**
**     params   required_bytes : amount of bytes to be allocated
**              alignment : valid alignments are values with a power of 2
**     return   pointer to allocated memory
*/
/*---------------------------------------------------------------------------*/
void* alloc_aligned_malloc(size_t required_bytes, size_t alignment)
{
    void* p1;  // initial block allocated by malloc
    void** p2; // aligned block to be returned

    // allocate enough memory with extra space to store the alignment offset and the size of the allocated block
    size_t offset = alignment - 1 + sizeof(void*) + sizeof(size_t);
    p1 = (void*)malloc(required_bytes + offset);
    if (p1 == NULL)
    {
        return NULL;
    }

    // align the pointer
    p2 = (void**)(((uintptr_t)(p1)+offset) & ~(alignment - 1));

    // store the original pointer and the size of the allocated block just before the aligned memory
    p2[-1] = p1;
    *((size_t*)p2 - 2) = required_bytes + offset;
    bytes_allocated += (required_bytes + offset);

    return p2;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: alloc_aligned_free
**
**     brief    Frees allocated number of bytes with alignment.
**
**     params   p: pointer to allocated memory
**
**     return   void
*/
/*---------------------------------------------------------------------------*/
void alloc_aligned_free(void* p)
{
    if (p != NULL)
    {
        // retrieve the size of the allocated block
        size_t allocated_size = *((size_t*)p - 2);

        // update the bytes_allocated counter
        bytes_allocated -= allocated_size;

        // free the original pointer
        free(((void**)p)[-1]);
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: alloc_get_bytes_allocated
**
**     brief    Get allocated number of bytes.
**
**     params   none
**
**     return   number of allocated bytes
*/
/*---------------------------------------------------------------------------*/
size_t alloc_get_bytes_allocated(void)
{
    return bytes_allocated;
}
