package org.ssketch;

import org.apache.spark.AccumulatorParam;

public class MatrixAccumulatorParam implements AccumulatorParam<double[][]> {


	public double[][] addInPlace(double[][] arg0, double[][] arg1) {
		int i,j;
		int rows=arg0.length;
		int cols=arg0[0].length;
		//double[][] res = new double[rows][cols];
		for(i=0; i< rows; i++)
		{
			for(j=0; j<cols; j++)
			{
				arg0[i][j] = arg0[i][j] + arg1[i][j];
			}
		}
		return arg0;
	}

	public double[][] zero(double[][] arg0) {
		return arg0;
	}

	public double[][] addAccumulator(double[][] arg0, double[][] arg1) {
		return addInPlace(arg0,arg1);
	}
	

}
