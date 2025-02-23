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

#ifndef __IO_H
#define __IO_H

#if defined(__cplusplus)
extern "C" {
#endif

#include "matrix.h"

// Structure to store the list of file names
typedef struct
{
    char *dir_path;       // directory path
    char **file_names;    // array of file names
    int file_count;       // number of files in the list
} FileList;

int       io_count_files_in_directory(const char *dir_path);
FileList* io_create_file_list(int nbr_files);
void      io_free_file_list(FileList *list);
void      io_populate_file_list(FileList *list, const char *dir_path);
void      io_print_file_list(FileList *list);
int       io_read_file_from_list(FileList *list, uint32_t index, matrix_t **data_set, uint32_t depth, uint32_t bytes_per_depth);
int       io_read_file(matrix_t **data_set, uint32_t depth, uint32_t bytes_per_depth, const char *full_path);

#if defined(__cplusplus)
}
#endif

#endif
