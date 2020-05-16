using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace test_project
{
    public abstract class SymmetryAnnotator
    {
        public List<Variable<double>> Vars;
        public Matrix<double> Constraints;
        public int NumVar;
        public int NumCons;

        public Matrix<double> GetConstraints()
        {
            return this.Constraints;
        }

        public Vector<double> GetNullSpace()
        {
            return this.Constraints.Kernel()[0];
        }

        protected void AddVariablesToConstraints(Variable<double>[] Variables)
        {
            int ExistingVars = this.NumVar;
            for (int i = 0; i < Variables.Length; i++)
            {
                bool exists = false;
                for (int j = 0; j < this.NumVar; j++)
                {
                    if (Object.ReferenceEquals(this.Vars[j], Variables[i]))
                    {
                        // Variable exists in constraint matrix
                        exists = true;
                        break;
                    }
                }
                if (!exists)
                {
                    this.Vars.Add(Variables[i]);
                    this.NumVar += 1;
                }
            }

            if (this.Constraints != null & this.NumVar != ExistingVars)
            {
                Matrix<double> NewColumns = Matrix<double>.Build.Dense(this.NumCons, this.NumVar - ExistingVars);
                this.Constraints = this.Constraints.Append(NewColumns);
            }
        }

        public abstract Variable<double> SumOp(Variable<double> v1, Variable<double> v2);

        public abstract Variable<double> ProductOp(Variable<double> v1, Variable<double> v2);

        public abstract void ObservedValueOp(Variable<double> v, double ObservedValue);
    }

    public class ScalingAnnotator : SymmetryAnnotator
    {
        public ScalingAnnotator()
        {
            this.Vars = new List<Variable<double>>();
            this.NumVar = 0;
            this.NumCons = 0;
        }

        public override Variable<double> SumOp(Variable<double> v1, Variable<double> v2)
        {
            Variable<double> v3 = v1 + v2;
            Variable<double>[] variables = {v1, v2, v3};

            this.AddVariablesToConstraints(variables);

            if (this.Constraints == null)
            {
                this.Constraints = DenseMatrix.OfArray(new double[2, this.NumVar]);
            }
            else
            {
                // Transpose, add new columns, transpose back
                Matrix<double> NewRows = DenseMatrix.OfArray(new double[this.NumVar, 2]);
                this.Constraints = this.Constraints.Transpose().Append(NewRows).Transpose();
            }
            this.NumCons += 2;

            // Create constraints d1 = d3, d2 = d3
            for (int i = 0; i < this.NumVar; i++)
            {
                if (Object.ReferenceEquals(this.Vars[i], v1))
                {
                    this.Constraints[this.NumCons - 2, i] = 1;
                    this.Constraints[this.NumCons - 1, i] = 0;
                }
                else if (Object.ReferenceEquals(this.Vars[i], v2))
                {
                    this.Constraints[this.NumCons - 2, i] = 0;
                    this.Constraints[this.NumCons - 1, i] = 1;
                }
                else if (Object.ReferenceEquals(Vars[i], v3))
                {
                    this.Constraints[this.NumCons - 2, i] = -1;
                    this.Constraints[this.NumCons - 1, i] = -1;
                }
                else
                {
                    this.Constraints[this.NumCons - 2, i] = 0;
                    this.Constraints[this.NumCons - 1, i] = 0;
                }
            }

            return v3;
        }

        public override Variable<double> ProductOp(Variable<double> v1, Variable<double> v2)
        {
            Variable<double> v3 = v1 * v2;
            Variable<double>[] variables = {v1, v2, v3};

            this.AddVariablesToConstraints(variables);

            if (this.Constraints == null)
            {
                this.Constraints = DenseMatrix.OfArray(new double[1, this.NumVar]);
            }
            else
            {
                // Transpose, add new columns, transpose back
                Matrix<double> NewRows = DenseMatrix.OfArray(new double[this.NumVar, 1]);
                this.Constraints = this.Constraints.Transpose().Append(NewRows).Transpose();
            }
            this.NumCons += 1;

            // Create constraint d1 + d2 = d3
            for (int i = 0; i < this.NumVar; i++)
            {
                if (Object.ReferenceEquals(this.Vars[i], v1) | Object.ReferenceEquals(this.Vars[i], v2))
                {
                    this.Constraints[this.NumCons - 1, i] = 1;
                }
                else if (Object.ReferenceEquals(Vars[i], v3))
                {
                    this.Constraints[this.NumCons - 1, i] = -1;
                }
                else
                {
                    this.Constraints[this.NumCons - 1, i] = 0;
                }
            }

            return v3;
        }

        public override void ObservedValueOp(Variable<double> v, double ObservedValue)
        {
            v.ObservedValue = ObservedValue;
            Variable<double>[] variables = {v};

            this.AddVariablesToConstraints(variables);

            if (this.Constraints == null)
            {
                this.Constraints = DenseMatrix.OfArray(new double[1, this.NumVar]);
            }
            else
            {
                // Transpose, add new columns, transpose back
                Matrix<double> NewRows = DenseMatrix.OfArray(new double[this.NumVar, 1]);
                this.Constraints = this.Constraints.Transpose().Append(NewRows).Transpose();
            }
            this.NumCons += 1;

            // Create constraint d = 0
            for (int i = 0; i < this.NumVar; i++)
            {
                if (Object.ReferenceEquals(this.Vars[i], v))
                {
                    this.Constraints[this.NumCons - 1, i] = 1;
                }
                else
                {
                    this.Constraints[this.NumCons - 1, i] = 0;
                }
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            SymmetryAnnotator sa = new ScalingAnnotator();

            Variable<double> t1 = Variable.GaussianFromMeanAndVariance(1, 1).Named("t1");
            Variable<double> t2 = Variable.GaussianFromMeanAndVariance(10, 1).Named("t2");
            Variable<double> t3 = sa.SumOp(t1, t2);
            Variable<double> t4 = Variable.GaussianFromMeanAndVariance(2, 1).Named("t4");

            // Variable<double> t5_sa = (t3 * t4).Named("t5_sa");
            Variable<double> t5 = ((t1 + t2) * t4).Named("t5");
            // Variable<double> t6 = sa.SumOp(t3, t4).Named("t6");
            Variable<double> t7 = sa.ProductOp(t3, t4).Named("t7");

            InferenceEngine engine = new InferenceEngine();
            
            // Gaussian t5_sa_post = engine.Infer<Gaussian>(t5_sa);
            Gaussian t5_posterior = engine.Infer<Gaussian>(t5);
            Gaussian t7_sa_post = engine.Infer<Gaussian>(t7);

            Console.WriteLine(sa.GetConstraints().ToString());
            Console.WriteLine(sa.GetNullSpace().ToString());

            // Console.WriteLine("Th5_sa Posterior: " + t5_sa_post);
            Console.WriteLine("Theta5 Posterior: " + t5_posterior);
            Console.WriteLine("Th7_sa Posterior: " + t7_sa_post);
            
            // t5.ObservedValue = 11;
            sa.ObservedValueOp(t7, 11);

            var t4_posterior = engine.Infer(t4); // Gaussian(1.055, 0.02293)

            Console.WriteLine("Theta4 Posterior: " + t4_posterior);

            Console.WriteLine(sa.GetConstraints().ToString());
            Console.WriteLine(sa.GetNullSpace().ToString());
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
