using System;
using System.Numerics;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Linq;
using MathNet.Numerics.IntegralTransforms;

namespace DigitalMusicAnalysis
{
    public class timefreq
    {
        public float[][] timeFreqData;
        public int wSamp;

        public timefreq(float[] x, int windowSamp)
        {
            Console.WriteLine($"Starting STFT processing for {x.Length} samples with window size {windowSamp}");
            System.Diagnostics.Debug.WriteLine($"Starting STFT processing for {x.Length} samples with window size {windowSamp}");
            var totalTimer = Stopwatch.StartNew();
            
            this.wSamp = windowSamp;

            timeFreqData = new float[wSamp/2][];

            int nearest = (int)Math.Ceiling((double)x.Length / (double)wSamp);
            nearest = nearest * wSamp;

            var paddingTimer = Stopwatch.StartNew();

            Complex[] compX = new Complex[nearest];
            
            // Sequential approach is faster than parallel ~45%
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
            paddingTimer.Stop();
            Console.WriteLine($"Zero padding (sequential) completed in {paddingTimer.ElapsedMilliseconds} ms");

            int cols = 2 * nearest /wSamp;

            for (int jj = 0; jj < wSamp / 2; jj++)
            {
                timeFreqData[jj] = new float[cols];
            }

            var stftTimer = Stopwatch.StartNew();
            timeFreqData = stft(compX, wSamp);
            stftTimer.Stop();
            Console.WriteLine($"STFT computation completed in {stftTimer.ElapsedMilliseconds} ms");
            
            totalTimer.Stop();
            Console.WriteLine($"Total timefreq constructor time: {totalTimer.ElapsedMilliseconds} ms");
        }

        float[][] stft(Complex[] x, int wSamp)
        {
            Console.WriteLine($"Computing STFT for {x.Length} samples");
            var stftTimer = Stopwatch.StartNew();
            
            int N = x.Length;
            float fftMax = 0;
            
            float[][] Y = new float[wSamp / 2][];

            for (int ll = 0; ll < wSamp / 2; ll++)
            {
                Y[ll] = new float[2 * (int)Math.Floor((double)N / (double)wSamp)];
            }
            
            Complex[] temp = new Complex[wSamp];
            Complex[] tempFFT = new Complex[wSamp];

            int windowCount = (int)(2 * Math.Floor((double)N / (double)wSamp) - 1);
            Console.WriteLine($"Processing {windowCount} overlapping windows");
            
            var windowTimer = Stopwatch.StartNew();

            //ConcurrentBag for thread safety
            var localMaxima = new System.Collections.Concurrent.ConcurrentBag<float>();
            
            Parallel.For(0, windowCount, ii =>
            {
                // Thread-local variables
                Complex[] localTemp = new Complex[wSamp];
                Complex[] localTempFFT = new Complex[wSamp];
                float threadLocalMax = 0.0f;
                
                if (ii % 100 == 0 && ii > 0)
                {
                    Console.WriteLine($"Processed {ii}/{windowCount} windows ({(ii * 100.0 / windowCount):F1}%)");
                }

                // Extract window
                for (int localJj = 0; localJj < wSamp; localJj++)
                {
                    localTemp[localJj] = x[ii * (wSamp / 2) + localJj];
                }

                // Clone array for FFT since Fourier.Forward modifies in place
                Complex[] fftInput = (Complex[])localTemp.Clone();
                Fourier.Forward(fftInput, FourierOptions.NoScaling);
                localTempFFT = fftInput;

                // Store results
                for (int localKk = 0; localKk < wSamp / 2; localKk++)
                {
                    float magnitude = (float)Complex.Abs(localTempFFT[localKk]);                        
                    Y[localKk][ii] = magnitude;

                    if (magnitude > threadLocalMax)
                    {
                        threadLocalMax = magnitude;
                    }
                }
                
                // Store thread-local max
                localMaxima.Add(threadLocalMax);
            });
            
            // Find global maximum after parallel section
            fftMax = localMaxima.Count > 0 ? localMaxima.Max() : 0.0f;
            windowTimer.Stop();
            Console.WriteLine($"Window processing completed in {windowTimer.ElapsedMilliseconds} ms");

            var normalizationTimer = Stopwatch.StartNew();

            // Parallelize normalization
            Parallel.For(0, windowCount, ii =>
            {
                for (int kk = 0; kk < wSamp / 2; kk++)
                {
                    Y[kk][ii] /= fftMax;
                }
            });
            normalizationTimer.Stop();
            Console.WriteLine($"Normalization completed in {normalizationTimer.ElapsedMilliseconds} ms");

            stftTimer.Stop();
            Console.WriteLine($"Total STFT time: {stftTimer.ElapsedMilliseconds} ms");
            
            return Y;
        }
        
    }
}
