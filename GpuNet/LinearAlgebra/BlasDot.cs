using System;
using System.Collections.Generic;
using System.Text;
using ILGPU;
using ILGPU.Runtime;

namespace GpuNet.LinearAlgebra
{
    public class BlasDot
    {
        public static void MultiplyPointwiseKernel(int n, ArrayView<double> xy, ArrayView<double> y)
        {
            int t = Grid.GlobalIndex.X;
            if (t < n)
            {
                xy[t] *= y[t];
            }
        }

        public static void ReduceSumKernel(int n, ArrayView<double> input, ArrayView<double> partialOutput)
        {
            // Copy part of the input elements into shared memory. Each thread copies 2 values.
            ArrayView<double> partialShared = SharedMemory.GetDynamic<double>();
            int t = Group.IdxX;
            int start = 2 * Grid.IdxX * Group.DimX;
            partialShared[t] = input[start + t];
            partialShared[Group.DimX + t] = input[start + Group.DimX + t];

            // If each thread copied 2 consecutive values, global memory access would not be coalesced.

            // Perform log(n) partial reductions in shared memory
            for (int stride = Group.DimX; stride > 0; stride /= 2)
            {
                Group.Barrier();
                if (t < stride)
                {
                    partialShared[t] += partialShared[t + stride];
                }
            }

            // The partial result is now stored at the start of shared memory. 
            // Only one thread copies it to global memory.
            if (t == 0)
            {
                partialOutput[Grid.IdxX] = partialShared[0];
            }
        }

        public static double DotProduct(Accelerator accelerator, int n, double[] x, double[] y)
        {
            // Initialize memory on device
            MemoryBuffer<double> deviceXY = accelerator.Allocate<double>(n);
            MemoryBuffer<double> deviceY = accelerator.Allocate<double>(n);

            // Copy memory from host to device
            int offsetHost = 0;
            int offsetDevice = 0;
            deviceXY.CopyFrom(x, offsetHost, offsetDevice, n);
            deviceY.CopyFrom(y, offsetHost, offsetDevice, n);

            // Determine grid & group dimensions for multiplication kernel
            int groupSize = Math.Min(128, accelerator.MaxNumThreadsPerGroup);
            int gridSize = (n - 1) / groupSize + 1;
            KernelConfig launchDimension = (gridSize, groupSize);

            // Compile and launch kernel on device for pointwise multiplication
            var kernel = accelerator.LoadStreamKernel<int, ArrayView<double>, ArrayView<double>>(MultiplyPointwiseKernel);
            kernel(launchDimension, n, deviceXY, deviceY);

            // Wait for kernel to finish on device
            accelerator.Synchronize();

            // Use an extra array where the partial results from each group will be written
            MemoryBuffer<double> partialSumsDevice = accelerator.Allocate<double>(gridSize);
            //partialSumsDevice.MemSetToZero();

            // Determine dynamic shared memory size and use half the threads per group
            var memConfig = SharedMemoryConfig.RequestDynamic<double>(2 * groupSize);
            groupSize /= 2;
            //gridSize = (n - 1) / groupSize + 1;
            launchDimension = new KernelConfig(gridSize, groupSize, memConfig);

            // Compile and launch kernel on device for reduction multiplication
            kernel = accelerator.LoadStreamKernel<int, ArrayView<double>, ArrayView<double>>(ReduceSumKernel);
            kernel(launchDimension, n, deviceXY, partialSumsDevice);

            // Wait for kernel to finish on device
            accelerator.Synchronize();

            // Copy the partial sums back to host (Optionally reduce again if there are a lot of partial sums)
            var partialSumsHost = new double[partialSumsDevice.Length];
            partialSumsDevice.CopyTo(partialSumsHost, offsetDevice, offsetHost, partialSumsDevice.Length);

            // Perform reduction in host
            double totalSum = 0.0;
            for (int i = 0; i < partialSumsHost.Length; i++)
            {
                totalSum += partialSumsHost[i];
            }

            // Clean up
            deviceXY.Dispose();
            deviceY.Dispose();
            partialSumsDevice.Dispose();

            return totalSum;
        }
    }
}
