using System;
using System.Numerics;
using System.Threading.Tasks;

namespace DigitalMusicAnalysis
{
    /// <summary>
    /// Class to perform time-frequency analysis on digital music data.
    /// </summary>
    /// <remarks>
    /// This class contains methods to calculate Short-Time Fourier Transform (STFT) 
    /// and other related functionalities.
    /// </remarks>
    public class timefreq
    {
        /// <value>
        /// Gets the time-frequency data.
        /// </value>
        public float[][] timeFreqData { get; private set; }

        public int wSamp;

        public Complex[] twiddles;

        /// <summary>
        /// Maximum degree of parallelism for parallel operations.
        /// </summary>
        private readonly int maxDegreeOfParallelism = 5;

        /// <summary>
        /// Initializes a new instance of the <see cref="timefreq"/> class.
        /// </summary>
        /// <param name="x">The input data.</param>
        /// <param name="windowSamp">The window sample size.</param>
        /// <remarks>
        /// This constructor initializes twiddles and performs STFT on the input data.
        /// </remarks>
        public timefreq(float[] x, int windowSamp)
        {
            int ii;
            double pi = 3.14159265;
            Complex i = Complex.ImaginaryOne;
            this.wSamp = windowSamp;
            twiddles = new Complex[wSamp];

            for (ii = 0; ii < wSamp; ii++)
            {
                double a = 2 * pi * ii / (double)wSamp;
                twiddles[ii] = Complex.Pow(Complex.Exp(-i), (float)a);
            }

            timeFreqData = new float[wSamp / 2][];

            int nearest = (int)Math.Ceiling((double)x.Length / (double)wSamp);
            nearest = nearest * wSamp;

            Complex[] compX = new Complex[nearest];
            for (int kk = 0; kk < nearest; kk++)
            {
                if (kk < x.Length)
                {
                    compX[kk] = x[kk];
                }
                else
                {
                    compX[kk] = Complex.Zero;
                }
            }

            int cols = 2 * nearest / wSamp;

            for (int jj = 0; jj < wSamp / 2; jj++)
            {
                timeFreqData[jj] = new float[cols];
            }

            timeFreqData = stft(compX, wSamp);
        }

        /// <summary>
        /// Performs Short-Time Fourier Transform (STFT) on the given data.
        /// </summary>
        /// <param name="x">The input data of complex numbers.</param>
        /// <param name="wSamp">The window sample size.</param>
        /// <returns>A 2D array representing the STFT results.</returns>
        /// <remarks>
        /// This method is crucial for the frequency analysis of the input data.
        /// </remarks>
        float[][] stft(Complex[] x, int wSamp)
        {
            int N = x.Length;
            float fftMax = 0;
            ParallelOptions parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };

            float[][] Y = new float[wSamp / 2][];
            object lockObject = new object();

            for (int ll = 0; ll < wSamp / 2; ll++)
            {
                Y[ll] = new float[2 * (int)Math.Floor((double)N / (double)wSamp)];
            }

            Parallel.For(0, 2 * (int)Math.Floor((double)N / (double)wSamp) - 1, parallelOptions, ii =>
            {
                Complex[] localTemp = new Complex[wSamp];
                Complex[] localTempFFT;

                for (int jj = 0; jj < wSamp; jj++)
                {
                    localTemp[jj] = x[ii * (wSamp / 2) + jj];
                }

                localTempFFT = fft(localTemp);

                for (int kk = 0; kk < wSamp / 2; kk++)
                {
                    Y[kk][ii] = (float)Complex.Abs(localTempFFT[kk]);

                    lock (lockObject)
                    {
                        if (Y[kk][ii] > fftMax)
                        {
                            fftMax = Y[kk][ii];
                        }
                    }
                }
            });

            Parallel.For(0, 2 * (int)Math.Floor((double)N / (double)wSamp) - 1, parallelOptions, ii =>
            {
                for (int kk = 0; kk < wSamp / 2; kk++)
                {
                    Y[kk][ii] /= fftMax;
                }
            });

            return Y;
        }

        /// <summary>
        /// Performs the Fast Fourier Transform (FFT) on the given data.
        /// </summary>
        /// <param name="a">The input array of complex numbers to be transformed.</param>
        /// <returns>An array of complex numbers after the FFT.</returns>
        public Complex[] fft(Complex[] a)
        {
            int n = a.Length;
            Complex[] A = BitReverseCopy(a);

            ParallelOptions options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };

            for (int s = 1; s <= Math.Log(n, 2); s++)
            {
                int m = (int)Math.Pow(2, s);
                Complex omegaM = Complex.Exp(-2.0 * Math.PI * Complex.ImaginaryOne / m);

                Parallel.For(0, n / m, options, kIndex =>
                {
                    int k = kIndex * m;
                    Complex omega = 1;
                    for (int j = 0; j < m / 2; j++)
                    {
                        Complex t = omega * A[k + j + m / 2];
                        Complex u = A[k + j];
                        A[k + j] = u + t;
                        A[k + j + m / 2] = u - t;
                        omega *= omegaM;
                    }
                });
            }
            return A;
        }

        /// <summary>
        /// Creates a new array where the elements of the given array are rearranged based on a bit-reversal of their indices.
        /// </summary>
        /// <param name="a">The input array of complex numbers to be rearranged.</param>
        /// <returns>A new array with elements rearranged according to bit-reversal of their indices.</returns>
        /// <remarks>
        /// Bit-reversal is a common operation in FFT algorithms. This method is essential for 
        /// rearranging the input data before applying the FFT.
        /// </remarks>
        private Complex[] BitReverseCopy(Complex[] a)
        {
            int n = a.Length;
            Complex[] A = new Complex[n];
            for (int k = 0; k < n; k++)
            {
                A[ReverseBits(k, (int)Math.Log(n, 2))] = a[k];
            }
            return A;
        }

        /// <summary>
        /// Reverses the bits of the given number.
        /// </summary>
        /// <param name="num">The number whose bits are to be reversed.</param>
        /// <param name="bitLength">The length of the number in bits.</param>
        /// <returns>The number after reversing its bits.</returns>
        /// <remarks>
        /// This method is a utility function used for bit-reversal operations, particularly in FFT algorithms.
        /// </remarks>
        private int ReverseBits(int num, int bitLength)
        {
            int result = 0;
            for (int i = 0; i < bitLength; i++)
            {
                result = (result << 1) | (num & 1);
                num >>= 1;
            }
            return result;
        }

    }
}
