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

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "types.h"
#include "io.h"

#define MAX_FILENAME 260  // maximum file name length (Windows limit)

// get the prefix (number) of the filename
int io_get_prefix_number_from_filename(char *filename)
{
    char filename_copy[MAX_FILENAME];
    int prefix_number;
    char *underscore_pos;

    strcpy(filename_copy, filename);
    underscore_pos = strchr(filename_copy, '_');

    if (underscore_pos != NULL)
    {
        // isolate the prefix
        *underscore_pos = '\0';

        // convert the prefix (before the underscore) to an integer
        prefix_number = atoi(filename_copy);
    }
    else
    {
        prefix_number = -1;
    }

    return prefix_number;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: io_count_files_in_directory
**
**     brief    Count the files in a directory.
**
**     params   dir_path: path to the directory
**     return   number of files in the directory
*/
/*---------------------------------------------------------------------------*/
int io_count_files_in_directory(const char *dir_path)
{
    WIN32_FIND_DATAA file_data;
    HANDLE hFind;
    char search_path[MAX_PATH];
    int file_count = 0;

    // build the search pattern
    snprintf(search_path, sizeof(search_path), "%s\\*", dir_path);

    // start searching
    hFind = FindFirstFileA(search_path, &file_data);
    if (hFind == INVALID_HANDLE_VALUE)
    {
        printf("Failed to open directory: %s\n", dir_path);
        return -1;
    }

    do
    {
        // skip directories
        if (!(file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        {
            file_count++;
        }
    }
    while (FindNextFileA(hFind, &file_data) != 0);

    // close the handle
    FindClose(hFind);

    return file_count;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: io_create_file_list
**
**     brief    Initialize an empty file list.
**
**     params   nbr_files: number of files in the directory
**     return   pointer to the created file list
*/
/*---------------------------------------------------------------------------*/
FileList* io_create_file_list(int nbr_files)
{
    FileList *list;

    list = (FileList*)malloc(sizeof(FileList));
    if (list != NULL)
    {
        list->dir_path = (char*)malloc(MAX_FILENAME * sizeof(char));
        list->file_names = (char**)malloc(nbr_files * sizeof(char*));
        if (list->file_names != NULL)
        {
            for (int i = 0; i < nbr_files; i++)
            {
                list->file_names[i] = (char*)malloc(MAX_FILENAME);
            }
            list->file_count = 0;
        }
        else
        {
            free(list->dir_path);
            free(list);
            list = NULL;
        }
    }
    return list;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: io_free_file_list
**
**     brief    Free the file list.
**
**     params   list: pointer to the file list
**     return   none
*/
/*---------------------------------------------------------------------------*/
void io_free_file_list(FileList *list)
{
    for (int i = 0; i < list->file_count; i++)
    {
        free(list->file_names[i]);
    }
    free(list->file_names);
    free(list->dir_path);
    free(list);
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: io_populate_file_list
**
**     brief    Populate the file list with file names from the directory.
**
**     params   list: pointer to the file list
**              dir_path: path to the directory
**     return   none
*/
/*---------------------------------------------------------------------------*/
void io_populate_file_list(FileList *list, const char *dir_path)
{
    WIN32_FIND_DATAA file_data;
    HANDLE hFind;
    char search_path[MAX_PATH];

    // build the search pattern
    snprintf(search_path, sizeof(search_path), "%s\\*", dir_path);

    hFind = FindFirstFileA(search_path, &file_data);
    if (hFind == INVALID_HANDLE_VALUE)
    {
        printf("Error: Could not open directory %s\n", dir_path);
        return;
    }

    // store directory path
    strncpy(list->dir_path, dir_path, MAX_FILENAME - 1);
    list->dir_path[MAX_FILENAME - 1] = '\0'; // ensure null-termination

    do
    {
        // skip directories
        if (!(file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        {
            strncpy(list->file_names[list->file_count], file_data.cFileName, MAX_FILENAME - 1);
            list->file_names[list->file_count][MAX_FILENAME - 1] = '\0'; // ensure null-termination
            list->file_count++;
        }
    }
    while (FindNextFileA(hFind, &file_data) != 0);

    FindClose(hFind);
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: io_print_file_list
**
**     brief    Print the file list.
**
**     params   list: pointer to the file list
**     return   none
*/
/*---------------------------------------------------------------------------*/
void io_print_file_list(FileList *list)
{
    printf("Files in the directory: %s\n", list->dir_path);
    for (int i = 0; i < list->file_count; i++)
    {
        printf("%d: %s\n", i + 1, list->file_names[i]);
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: io_read_rgb_file_from_list
**
**     brief    Read a specific raw data file from the list in binary mode.
**
**     params   list: pointer to the file list
**              index: index of the file in the list
**              data_set: data channels (depth) where to store all data of the raw data file
**              depth: number of channels
**              bytes_per_depth: number of bytes per channel (depth)
**     return   first characters (number) of the file name (class label)
*/
/*---------------------------------------------------------------------------*/
int io_read_file_from_list(FileList *list, uint32_t index, matrix_t **data_set, uint32_t depth, uint32_t bytes_per_depth)
{
    FILE *file;
    char file_path[MAX_PATH];
    uint32_t file_size;
    uint32_t data_count;
    uint32_t single_data_size = depth * bytes_per_depth;
    uint8_t data_bytes[4];
    uint32_t data_word;
    uint32_t normalize_factor;
    int ret = -1;

    if (depth == 0ul)
    {
        printf("Error: Invalid depth. Valid range is greater than 0.\n");
        return ret;
    }

    if ((bytes_per_depth == 0ul) || (bytes_per_depth > 4ul))
    {
        printf("Error: Invalid bytes per depth. Valid range is 1 to 4.\n");
        return ret;
    }

    if (index >= ((uint32_t)list->file_count))
    {
        printf("Error: Invalid file index %d. Valid range is 0 to %d.\n", (int)index, list->file_count - 1);
        return ret;
    }

    // construct the full file path
    snprintf(file_path, sizeof(file_path), "%s\\%s", list->dir_path, list->file_names[index]);

    file = fopen(file_path, "rb");
    if (file == NULL)
    {
        printf("Error: Could not open file %s\n", file_path);
        return ret;
    }

    // get the file size
    fseek(file, 0, SEEK_END);
    file_size = (uint32_t)ftell(file);
    rewind(file);

    // check if file size is dividable by single_data_size
    if ((file_size % single_data_size) != 0ul)
    {
        printf("Error: File size is not dividable by %d.\n", (int)single_data_size);
        fclose(file);
        return ret;
    }

    data_count = file_size / single_data_size;

    // read data
    for (uint32_t c = 0ul; c < data_count; c++)
    {
        for (uint32_t d = 0ul; d < depth; d++)
        {
            if (fread(data_bytes, 1, bytes_per_depth, file) == bytes_per_depth)
            {
                data_word = 0ul;
                normalize_factor = 0ul;
                for (uint32_t b = 0ul; b < bytes_per_depth; b++)
                {
                    data_word <<= 8;
                    data_word |= (uint32_t)data_bytes[b];
                    normalize_factor <<= 8;
                    normalize_factor |= 255ul;
                }
                data_set[d]->values[c] = (real)data_word / (real)normalize_factor;
            }
            else
            {
                printf("Error: Unexpected end of file while reading data.\n");
                break;
            }
        }
    }

    fclose(file);

    ret = io_get_prefix_number_from_filename(list->file_names[index]);
    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: io_read_file
**
**     brief    Read a specific raw data file in binary mode.
**
**     params   data_set: data channels (depth) where to store all data of the raw data file
**              depth: number of channels
**              bytes_per_depth: number of bytes per channel (depth)
**              full_path: full path to the file
**     return   0 on success, -1 on failure
*/
/*---------------------------------------------------------------------------*/
int io_read_file(matrix_t **data_set, uint32_t depth, uint32_t bytes_per_depth, const char *full_path)
{
    FILE *file;
    uint32_t file_size;
    uint32_t data_count;
    uint32_t single_data_size = depth * bytes_per_depth;
    uint8_t data_bytes[4];
    uint32_t data_word;
    uint32_t normalize_factor;
    int ret = -1;

    if (depth == 0ul)
    {
        printf("Error: Invalid depth. Valid range is greater than 0.\n");
        return ret;
    }

    if ((bytes_per_depth == 0ul) || (bytes_per_depth > 4ul))
    {
        printf("Error: Invalid bytes per depth. Valid range is 1 to 4.\n");
        return ret;
    }

    file = fopen(full_path, "rb");
    if (file == NULL)
    {
        printf("Error: Could not open file %s\n", full_path);
        return ret;
    }

    // get the file size
    fseek(file, 0, SEEK_END);
    file_size = (uint32_t)ftell(file);
    rewind(file);

    // check if file size is divisible by single_data_size
    if ((file_size % single_data_size) != 0ul)
    {
        printf("Error: File size is not divisible by %d.\n", (int)single_data_size);
        fclose(file);
        return ret;
    }

    data_count = file_size / single_data_size;

    // read data
    for (uint32_t c = 0ul; c < data_count; c++)
    {
        for (uint32_t d = 0ul; d < depth; d++)
        {
            if (fread(data_bytes, 1, bytes_per_depth, file) == bytes_per_depth)
            {
                data_word = 0ul;
                normalize_factor = 0ul;
                for (uint32_t b = 0ul; b < bytes_per_depth; b++)
                {
                    data_word <<= 8;
                    data_word |= (uint32_t)data_bytes[b];
                    normalize_factor <<= 8;
                    normalize_factor |= 255ul;
                }
                data_set[d]->values[c] = (real)data_word / (real)normalize_factor;
            }
            else
            {
                printf("Error: Unexpected end of file while reading data.\n");
                break;
            }
        }
    }

    fclose(file);

    ret = 0;
    return ret;
}
