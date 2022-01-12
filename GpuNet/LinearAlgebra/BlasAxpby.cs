using System;
using System.Collections.Generic;
using System.Text;
using ILGPU;
using ILGPU.Runtime;

namespace GpuNet.LinearAlgebra
{
    public class BlasAxpby
    {
        public static void AxpbyKernel(int n, double a, ArrayView<double> x, double b, ArrayView<double> y,
            ArrayView<double> result)
        {
            int t = Grid.GlobalIndex.X;
            if (t < n) // Possible warp divergence. Unavoidable.
            {
                result[t] = a * x[t] + b * y[t];
            }
        }

        public static void Axpby(Accelerator accelerator, int n, double a, double[] x, double b, double[] y, double[] result)
        {
            // Initialize memory on device
            MemoryBuffer<double> deviceX = accelerator.Allocate<double>(n);
            MemoryBuffer<double> deviceY = accelerator.Allocate<double>(n);
            MemoryBuffer<double> deviceResult = accelerator.Allocate<double>(n);

            // Copy memory from host to device
            int offsetHost = 0;
            int offsetDevice = 0;
            deviceX.CopyFrom(x, offsetHost, offsetDevice, n);
            deviceY.CopyFrom(y, offsetHost, offsetDevice, n);
            deviceResult.CopyFrom(result, offsetHost, offsetDevice, n);

            // Determine grid & group dimensions
            int groupSize = Math.Min(1024, accelerator.MaxNumThreadsPerGroup);
            int gridSize = (n - 1) / groupSize + 1;
            KernelConfig launchDimension = (gridSize, groupSize);

            // Compile and launch kernel on device
            var kernel = accelerator.LoadStreamKernel<int, double, ArrayView<double>, double, ArrayView<double>,
                ArrayView<double>>(AxpbyKernel);
            kernel(launchDimension, n, a, deviceX, b, deviceY, deviceResult);

            // Wait for kernel to finish on device
            accelerator.Synchronize();

            // Copy result from device to host
            deviceResult.CopyTo(result, offsetDevice, offsetHost, n);

            // Clean up
            deviceX.Dispose();
            deviceY.Dispose();
            deviceResult.Dispose();
        }
    }
}
