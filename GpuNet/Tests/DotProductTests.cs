using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using GpuNet.LinearAlgebra;
using ILGPU;
using ILGPU.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GpuNet.Tests
{
    public static class DotProductTests
    {
        [Fact]
        public static void TestDotProduct()
        {
            int n = 1024;
            double[] x = CreateRandomVector(11, n);
            double[] y = CreateRandomVector(23, n);
            double expectedDotProduct = DotProductSerial(x, y);

            // Create main context
            using (var context = new Context())
            {
                // For each available accelerator...
                foreach (var acceleratorId in Accelerator.Accelerators)
                {
                    // Create default accelerator for the given accelerator id
                    using (var accelerator = Accelerator.Create(context, acceleratorId))
                    {
                        if (accelerator.AcceleratorType != AcceleratorType.Cuda) continue;

                        Debug.WriteLine($"Performing operations on {accelerator}");
                        double computedDotProduct = BlasDot.DotProduct(accelerator, n, x, y);
                        Assert.Equal(expectedDotProduct, computedDotProduct, 4);
                    }
                }
            }
        }

        public static double[] X => new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8};

        public static double[] Y => new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        public static double[] CreateRandomVector(int seed, int length)
        {
            var random = new Random(seed);
            double[] vector = new double[length];
            for (int i = 0; i < length; i++)
            {
                vector[i] = random.NextDouble();
            }
            return vector;
        }

        public static double DotProductSerial(double[] x, double[] y)
        {
            int n = x.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += x[i] * y[i];
            }
            return sum;
        }
    }
}
