/* Copyright (C) 2021 Free Software Foundation, Inc.
   Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CL_TARGET_OPENCL_VERSION 220

#include <CL/opencl.h>

#define LANE_NUM 32
#define BUFF_SIZE 1000
#define QUARY_SIZE 5
#define PLATFORM_ID 0
#define DEVICE_ID 0

#define MEM_SIZE (1024)
#define MAX_BINARY_SIZE (0x100000)

int main(int argc, char *argv[])
{
  cl_platform_id sPlatforms[QUARY_SIZE];
  cl_uint iNumPlatforms = 0;
  cl_device_id sDevices[QUARY_SIZE];
  cl_uint iNumDevices = 0;
  cl_context sContext;
  cl_program sProgram;
  cl_command_queue sQueue;
  cl_kernel sKernel;
  cl_event sKernelEnd;
  size_t sWorkSize[1] = { LANE_NUM };
  cl_mem sOutBuff, sInBuff;
  unsigned int i;
  char buffer[BUFF_SIZE];
  cl_int _err;

  FILE *fp;
  size_t binary_size;
  char *binary_buf;
  cl_int binary_status;

  /* Load kernel binary */
  fp = fopen("./outputs/gdb.rocm/opencl/lane-pc-vega20/lane-pc-vega20-kernel.so", "r");

  if (!fp)
    {
      /* Try on current directory */
      fp = fopen("./lane-pc-vega20-kernel.so", "r");

      if (!fp)
	return 1;
    }

  binary_buf = (char *)malloc(MAX_BINARY_SIZE);
  binary_size = fread(binary_buf, 1, MAX_BINARY_SIZE, fp);

  fclose(fp);

  clGetPlatformIDs(QUARY_SIZE, sPlatforms, &iNumPlatforms);

  printf("Platforms found: %d\n", iNumPlatforms);

  if (iNumPlatforms == 0) return 1;

  clGetPlatformInfo(sPlatforms[PLATFORM_ID], CL_PLATFORM_VENDOR,
		    BUFF_SIZE, buffer, NULL);
  printf("Vendor= %s\n", buffer);
  clGetPlatformInfo(sPlatforms[PLATFORM_ID], CL_PLATFORM_VERSION,
		    BUFF_SIZE, buffer, NULL);
  printf("Version= %s\n", buffer);
  clGetPlatformInfo(sPlatforms[PLATFORM_ID], CL_PLATFORM_NAME,
		    BUFF_SIZE, buffer, NULL);
  printf("Name= %s\n", buffer);

  clGetDeviceIDs(sPlatforms[PLATFORM_ID], CL_DEVICE_TYPE_GPU,
		 QUARY_SIZE, sDevices, &iNumDevices);

  printf("Devices found: %d\n", iNumDevices);

  if (iNumDevices == 0) return 1;

  clGetDeviceInfo(sDevices[DEVICE_ID], CL_DEVICE_NAME,
		  sizeof(buffer), buffer, NULL);
  printf("\t\tName= %s\n", buffer);
  clGetDeviceInfo(sDevices[DEVICE_ID], CL_DEVICE_VENDOR,
		  sizeof(buffer), buffer, NULL);
  printf("\t\tVendor= %s\n", buffer);
  clGetDeviceInfo(sDevices[DEVICE_ID], CL_DEVICE_VERSION,
		  sizeof(buffer), buffer, NULL);
  printf("\t\tVersion= %s\n", buffer);

  sContext = clCreateContext(NULL, iNumDevices, sDevices, NULL, NULL, &_err);

  sProgram = clCreateProgramWithBinary(sContext, iNumDevices,
				       sDevices, (const size_t *)&binary_size,
  (const unsigned char **)&binary_buf, &binary_status, &_err);

  if (clBuildProgram(sProgram, iNumDevices,
		     sDevices, "", NULL, NULL) != CL_SUCCESS)
    {
      cl_int ret = clGetProgramBuildInfo(sProgram, sDevices[DEVICE_ID],
					 CL_PROGRAM_BUILD_LOG, sizeof(buffer),
					 buffer, NULL);
      fprintf(stderr, "Build failed %d :\n%s", ret, buffer);
      return 1;
    }

  sOutBuff = clCreateBuffer(sContext, CL_MEM_WRITE_ONLY,
			    sizeof(float)*LANE_NUM, NULL, &_err);
  sInBuff = clCreateBuffer(sContext, CL_MEM_READ_ONLY,
			   sizeof(int)*LANE_NUM, NULL, &_err);

  sKernel = clCreateKernel(sProgram, "AddrClassTest", &_err);
  clSetKernelArg(sKernel, 0, sizeof(sInBuff), &sInBuff);
  clSetKernelArg(sKernel, 1, sizeof(sOutBuff), &sOutBuff);

  sQueue = clCreateCommandQueueWithProperties(sContext,
					      sDevices[DEVICE_ID], 0, &_err);

  for (i = 0; i < LANE_NUM; i++)
    clEnqueueWriteBuffer(sQueue, sInBuff, CL_TRUE, i*sizeof(int),
			 sizeof(int), &i, 0, NULL, NULL);

  clEnqueueNDRangeKernel(sQueue, sKernel, 1, NULL, sWorkSize,
			 NULL, 0, NULL, &sKernelEnd);
  clWaitForEvents(1, &sKernelEnd);
  clReleaseEvent(sKernelEnd);

  printf("Results:");

  for (i = 0; i < LANE_NUM; i++)
    {
      int iTemp;
      clEnqueueReadBuffer(sQueue, sOutBuff, CL_TRUE,
			  2*sizeof(int) + i*sizeof(int), sizeof(int),
			  &iTemp, 0, NULL, NULL);
      printf("\nLane %d: %d", i, iTemp);
    }

  printf("\n");

  clReleaseMemObject(sOutBuff);
  clReleaseKernel(sKernel);
  clReleaseProgram(sProgram);
  clReleaseContext(sContext);
  return 0;
}
