using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace test_project
{
    class Program
    {
        static void Main(string[] args)
        {
            Variable<double> t1 = Variable.GaussianFromMeanAndVariance(1, 1).Named("t1");
            Variable<double> t2 = Variable.GaussianFromMeanAndVariance(10, 1).Named("t2");
            Variable<double> t4 = Variable.GaussianFromMeanAndVariance(2, 1).Named("t4");

            Variable<double> t5 = ((t1 + t2) * t4).Named("t5");

            InferenceEngine engine = new InferenceEngine();
            
            Gaussian t5_posterior = engine.Infer<Gaussian>(t5);

            Console.WriteLine("Theta5 Posterior: " + t5_posterior);

            t5.ObservedValue = 11;

            var t4_posterior = engine.Infer(t4);

            Console.WriteLine("Theta4 Posterior: " + t4_posterior);
        }

        // Coin example
        /*
        private void Main(string [] args)
        {
            Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
            Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
            Variable<bool> bothHeads = (firstCoin & secondCoin).Named("bothHeads");

            InferenceEngine ie = new InferenceEngine();
            Bernoulli query1 = (Bernoulli)ie.Infer(bothHeads);

            bothHeads.ObservedValue = false;
            Bernoulli query2 = (Bernoulli)ie.Infer(firstCoin);

            Console.WriteLine("Probability that both coin are heads: " + query1);
            Console.WriteLine("Probability of heads for first coin: " + query2);
        }
        */
    }
}
