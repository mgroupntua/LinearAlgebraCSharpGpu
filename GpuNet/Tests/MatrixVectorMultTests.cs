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
    public static class MatrixVectorMultTests
    {
        [Fact]
        public static void TestMatrixVectorMult()
        {
            int m = 2000, n = 1000;
            double[,] A = CreateRandomMatrix(11, m, n);
            double[] x = CreateRandomVector(23, n);
            double[] expectedAx = MvSerial(A, x);
            double[] computedAx = new double[m];

            // Create main context
            using (var context = new Context())
            {
                foreach (var acceleratorId in Accelerator.Accelerators)
                {
                    // Only for GPU accelerators
                    if (acceleratorId.AcceleratorType == AcceleratorType.CPU) continue;

                    // Create default accelerator for the given accelerator id
                    using (var accelerator = Accelerator.Create(context, acceleratorId))
                    {
                        Debug.WriteLine($"Performing operations on {accelerator}");
                        BlasMV.MatrixVectorMultiply(accelerator, m, n, A, x, computedAx);
                        AssertEqual(expectedAx, computedAx, 4);
                    }
                }
            }
        }

        public static double[] X => new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8};

        public static double[] Y => new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        public static double[,] CreateRandomMatrix(int seed, int numRows, int numCols)
        {
            var random = new Random(seed);
            double[,] matrix = new double[numRows, numCols];
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    matrix[i, j] = random.NextDouble();
                }
            }
            return matrix;
        }

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

        public static double[] MvSerial(double[,] A, double[] x)
        {
            int m = A.GetLength(0);
            int n = A.GetLength(1);
            var result = new double[m];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    result[i] += A[i, j] * x[j];
                }
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
