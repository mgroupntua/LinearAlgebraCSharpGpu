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
    public static class AxpbyTests
    {
        [Fact]
        public static void TestAxpby()
        {
            int n = 10000;
            double[] x = CreateRandomVector(11, n);
            double[] y = CreateRandomVector(23, n);
            double[] expected2x3y = AxpbySerial(2, x, 3, y);
            double[] computed2x3y = new double[n];

            // Create main context
            using (var context = new Context())
            {
                // For each available accelerator...
                foreach (var acceleratorId in Accelerator.Accelerators)
                {
                    // Create default accelerator for the given accelerator id
                    using (var accelerator = Accelerator.Create(context, acceleratorId))
                    {
                        Debug.WriteLine($"Performing operations on {accelerator}");
                        BlasAxpby.Axpby(accelerator, n, 2, x, 3, y, computed2x3y);
                        AssertEqual(expected2x3y, computed2x3y, 4);
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

        public static double[] AxpbySerial(double a, double[] x, double b, double[] y)
        {
            int n = x.Length;
            var result = new double[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = a * x[i] + b * y[i];
            }
            return result;
        }

        public static void AssertEqual(double[] expected, double[] actual, int precision)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], precision);
            }
        }
    }
}
