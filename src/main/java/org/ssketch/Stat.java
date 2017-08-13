package org.ssketch;

import java.util.ArrayList;

public class Stat {
	public String appName=null;
	public String dataSet=null;
	public int nRows=0;
	public int nCols=0;
	public int nIter=0;
	public double totalRunTime=0;
	public double preprocessTime=0;
	public double avgSketchTime=0;
	public double kSingularValue=0;
	public ArrayList<Double> sketchTime=new ArrayList<Double>();
	public ArrayList<Double> errorList=new ArrayList<Double>();
	Stat(){
		
	}
	
}
