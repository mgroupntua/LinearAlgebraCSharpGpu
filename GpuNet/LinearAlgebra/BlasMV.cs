using System;
using System.Collections.Generic;
using System.Text;
using ILGPU;
using ILGPU.Runtime;

namespace GpuNet.LinearAlgebra
{
    public class BlasMV
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="numRows">The number of rows</param>
        /// <param name="numCols">The number of colums</param>
        /// <param name="A">A (m x n) matrix.</param>
        /// <param name="x">The (n x 1) input (left hand side) vector</param>
        /// <param name="y">The (m x 1) output (right hand side) vector. Must be zero before calling the method.</param>
        public static void MvKernel(int numRows, int numCols, int groupWidth,
            ArrayView2D<double> A, ArrayView<double> x, ArrayView<double> y)
        {
            int groupHeight = Group.DimX;

            // Limit group width so that it does not exceed the number of columns
            int numColsPerThread;
            if ((Grid.IdxX + 1) * groupWidth <= numCols) numColsPerThread = groupWidth;
            else numColsPerThread = numCols % groupWidth;

            int colStart = Grid.IdxX * groupWidth; // the first column for this group

            // Copy the appropriate section of the left hand side vector into shared memory
            ArrayView<double> xShared = SharedMemory.GetDynamic<double>();
            if (Group.IdxX < numColsPerThread)
            {
                xShared[Group.IdxX] = x[colStart + Group.IdxX];
            }
            Group.Barrier(); // Wait for the threads that copy memory to finish

            // Initialize variable to hold the dot product section of this thread 
            double partialSum = 0;

            // Find the row of this thread. Note that threads with same local X index of groups 
            // with the same Y index, correspond to the same row.
            int row = Grid.IdxY * groupHeight + Group.IdxX;

            // The last threads of the bottom groups may be outside the matrix.
            if (row < numRows)
            {
                // Calculate the dot product section of this thread 
                for (int j = 0; j < numColsPerThread; j++)
                {
                    int col = colStart + j; // The column of A currently being processed by this thread
                    partialSum += A[row, col] * xShared[j];
                }

                // Then add it into the output vector. 
                // Warning: The output vector is in global memory and all threads have simultaneous access to it. 
                // The threads with same local X index of groups with the same Y index, correspond to the same 
                // row of the matrix and thus will try to write to the same index of the the output vector. 
                // To avoid race conditions, use atomics.
                Atomic.Add(ref y[row], partialSum);
            }
        }

        public static void MatrixVectorMultiply(Accelerator accelerator, int m, int n, double[,] A, double[] x, double[] y)
        {
            // Initialize memory on device
            MemoryBuffer2D<double> deviceA = accelerator.Allocate<double>(m, n);
            MemoryBuffer<double> deviceX = accelerator.Allocate<double>(n);
            MemoryBuffer<double> deviceY = accelerator.Allocate<double>(m);

            // Copy memory from host to device
            int offsetHost1D = 0;
            int offsetDevice1D = 0;
            deviceX.CopyFrom(x, offsetHost1D, offsetDevice1D, n);

            Index2 offsetHost2D = (0, 0);
            Index2 offsetDevice2D = (0, 0);
            Index2 extent = (m, n);
            deviceA.CopyFrom(A, offsetHost2D, offsetDevice2D, extent);

            // Set the output device vector to 0
            deviceY.MemSetToZero();

            // Determine grid & group dimensions for multiplication kernel
            int groupHeight = accelerator.MaxNumThreadsPerGroup;
            int groupWidth = 64; // must be < groupHeight
            int gridSizeX = (n - 1) / groupWidth + 1;
            int gridSizeY = (m - 1) / groupHeight + 1;
            Index2 groupSize = (groupHeight, 1);
            Index2 gridSize = (gridSizeX, gridSizeY);

            // Also determine dynamic shared memory size
            var memConfig = SharedMemoryConfig.RequestDynamic<double>(groupWidth);
            KernelConfig launchDimension = new KernelConfig(gridSize, groupSize, memConfig);

            // Compile and launch kernel on device
            var kernel = accelerator.LoadStreamKernel<int, int, int, ArrayView2D<double>, ArrayView<double>, ArrayView<double>>(
                MvKernel);
            kernel(launchDimension, m, n, groupWidth, deviceA, deviceX, deviceY);

            // Wait for kernel to finish on device
            accelerator.Synchronize();

            // Copy result from device to host
            deviceY.CopyTo(y, offsetDevice1D, offsetHost1D, m);

            // Clean up
            deviceA.Dispose();
            deviceX.Dispose();
            deviceY.Dispose();
        }
    }
}
