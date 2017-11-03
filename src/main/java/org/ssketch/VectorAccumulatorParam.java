package org.ssketch;

import org.apache.spark.AccumulatorParam;

/**
 * This class supports Accumulator of type double[]. It implements an element-by-element add operation for
 * two double[]
 *
 */
public class VectorAccumulatorParam implements AccumulatorParam<double[]> {



	public double[] addInPlace(double[] arg0, double[] arg1) {
		for(int i=0; i< arg0.length; i++)
		{
			arg0[i] += arg1[i];
		}
		return arg0;
	}
	public double[] zero(double[] arg0) {
		return arg0;
	}

	public double[] addAccumulator(double[] arg0, double[] arg1) {
		return addInPlace(arg0, arg1);
	}

}
