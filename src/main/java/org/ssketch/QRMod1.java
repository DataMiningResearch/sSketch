package org.ssketch;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.IntWritable;
import org.apache.log4j.Level;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.QRDecomposition;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.storage.StorageLevel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ssketch.FileFormat.OutputFormat;

import scala.Tuple2;


public class QRMod1 implements Serializable {

	private final static Logger log = LoggerFactory.getLogger(QRMod1.class);// getLogger(SparkPCA.class);

	static String dataset = "Untitled";
	static long startTime, endTime, totalTime;
	public static int nClusters = 4;
	public static Stat stat = new Stat();

	public static void main(String[] args) throws FileNotFoundException {
		org.apache.log4j.Logger.getLogger("org").setLevel(Level.ERROR);
		org.apache.log4j.Logger.getLogger("akka").setLevel(Level.ERROR);

		// Parsing input arguments
		final String inputPath;
		final String outputPath;
		final int nRows;
		final int nCols;
		final int nPCs;
		final int q;// default
		final double k_plus_one_singular_value;
		final double tolerance;
		final int subsample;
		final int subsampleNorm;

		try {
			inputPath = System.getProperty("i");
			if (inputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			printLogMessage("i");
			return;
		}
		try {
			outputPath = System.getProperty("o");
			if (outputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			printLogMessage("o");
			return;
		}

		try {
			nRows = Integer.parseInt(System.getProperty("rows"));
		} catch (Exception e) {
			printLogMessage("rows");
			return;
		}

		try {
			nCols = Integer.parseInt(System.getProperty("cols"));
		} catch (Exception e) {
			printLogMessage("cols");
			return;
		}

		try {
			k_plus_one_singular_value = Double.parseDouble(System.getProperty("SingularValue"));
		} catch (Exception e) {
			printLogMessage("SingularValue");
			return;
		}

		try {
			tolerance = Double.parseDouble(System.getProperty("tolerance"));
		} catch (Exception e) {
			printLogMessage("tolerance");
			return;
		}

		try {
			subsample = Integer.parseInt(System.getProperty("subSample"));
			System.out.println("Subsample is set to" + subsample);
		} catch (Exception e) {
			printLogMessage("subsample");
			return;
		}

		try {
			subsampleNorm = Integer.parseInt(System.getProperty("subSampleNorm"));
			System.out.println("SubsampleNorm is set to" + subsampleNorm);
		} catch (Exception e) {
			printLogMessage("subsampleNorm");
			return;
		}

		try {
			q = Integer.parseInt(System.getProperty("q"));
			System.out.println("No of q is set to" + q);
		} catch (Exception e) {
			printLogMessage("q");
			return;
		}

		try {

			if (Integer.parseInt(System.getProperty("pcs")) == nCols) {
				nPCs = nCols - 1;
				System.out
						.println("Number of princpal components cannot be equal to number of dimension, reducing by 1");
			} else
				nPCs = Integer.parseInt(System.getProperty("pcs"));
		} catch (Exception e) {
			printLogMessage("pcs");
			return;
		}
		/**
		 * Defaults for optional arguments
		 */
		int maxIterations = 50;
		OutputFormat outputFileFormat = OutputFormat.DENSE;
		int calculateError = 0;

		try {
			nClusters = Integer.parseInt(System.getProperty("clusters"));
			System.out.println("No of partition is set to" + nClusters);
		} catch (Exception e) {
			log.warn("Cluster size is set to default: " + nClusters);
		}

		try {
			maxIterations = Integer.parseInt(System.getProperty("maxIter"));
		} catch (Exception e) {
			log.warn("maximum iterations is set to default: maximum	Iterations=" + maxIterations);
		}

		try {
			dataset = System.getProperty("dataset");
		} catch (IllegalArgumentException e) {
			log.warn("Invalid Format " + System.getProperty("outFmt") + ", Default name for dataset" + dataset
					+ " will be used ");
		} catch (Exception e) {
			log.warn("Default oname for dataset " + dataset + " will be used ");
		}

		try {
			outputFileFormat = OutputFormat.valueOf(System.getProperty("outFmt"));
		} catch (IllegalArgumentException e) {
			log.warn("Invalid Format " + System.getProperty("outFmt") + ", Default output format" + outputFileFormat
					+ " will be used ");
		} catch (Exception e) {
			log.warn("Default output format " + outputFileFormat + " will be used ");
		}

		try {
			calculateError = Integer.parseInt(System.getProperty("calculateError"));
		} catch (Exception e) {
			log.warn(
					"Projected Matrix will not be computed, the output path will contain the principal components only");
		}

		// Setting Spark configuration parameters
		SparkConf conf = new SparkConf().setAppName("QRMod1");//.setMaster("local[*]");//
		// TODO
		// remove
		// this
		// part
		// for
		// building
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
		conf.set("spark.kryoserializer.buffer.max", "128m");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// compute principal components
		computePrincipalComponents(sc, inputPath, outputPath, nRows, nCols, nPCs, subsample, tolerance,
				k_plus_one_singular_value, q, maxIterations, calculateError, subsampleNorm);

		// log.info("Principal components computed successfully ");
	}

	/**
	 * Compute principal component analysis where the input is a path for a
	 * hadoop sequence File <IntWritable key, VectorWritable value>
	 * 
	 * @param sc
	 *            Spark context that contains the configuration parameters and
	 *            represents connection to the cluster (used to create RDDs,
	 *            accumulators and broadcast variables on that cluster)
	 * @param inputPath
	 *            Path to the sequence file that represents the input matrix
	 * @param nRows
	 *            Number of rows in input Matrix
	 * @param nCols
	 *            Number of columns in input Matrix
	 * @param nPCs
	 *            Number of desired principal components
	 * @param errRate
	 *            The sampling rate that is used for computing the
	 *            reconstruction error
	 * @param maxIterations
	 *            Maximum number of iterations before terminating
	 * @return Matrix of size nCols X nPCs having the desired principal
	 *         components
	 * @throws FileNotFoundException
	 */
	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, String inputPath,
			String outputPath, final int nRows, final int nCols, final int nPCs, final int subsample,
			final double tolerance, final double k_plus_one_singular_value, final int q, final int maxIterations,
			final int calculateError, final int subsampleNorm) throws FileNotFoundException {

		/**
		 * preprocess the data
		 * 
		 * @param nClusters
		 * 
		 */
		startTime = System.currentTimeMillis();

		// Read from sequence file
		JavaPairRDD<IntWritable, VectorWritable> seqVectors = sc.sequenceFile(inputPath, IntWritable.class,
				VectorWritable.class, nClusters);

		JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors = seqVectors
				.map(new Function<Tuple2<IntWritable, VectorWritable>, org.apache.spark.mllib.linalg.Vector>() {

					public org.apache.spark.mllib.linalg.Vector call(Tuple2<IntWritable, VectorWritable> arg0)
							throws Exception {

						org.apache.mahout.math.Vector mahoutVector = arg0._2.get();
						Iterator<Element> elements = mahoutVector.nonZeroes().iterator();
						ArrayList<Tuple2<Integer, Double>> tupleList = new ArrayList<Tuple2<Integer, Double>>();
						while (elements.hasNext()) {
							Element e = elements.next();
							if (e.index() >= nCols || e.get() == 0)
								continue;
							Tuple2<Integer, Double> tuple = new Tuple2<Integer, Double>(e.index(), e.get());
							tupleList.add(tuple);
						}
						org.apache.spark.mllib.linalg.Vector sparkVector = Vectors.sparse(nCols, tupleList);
						return sparkVector;
					}
				}).persist(StorageLevel.MEMORY_ONLY_SER()); // TODO
															// change
															// later;

		// 1. Mean Job : This job calculates the mean and span of the columns of
		// the input RDD<org.apache.spark.mllib.linalg.Vector>
		final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[nCols], new VectorAccumulatorParam());
		final double[] internalSumY = new double[nCols];
		vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

			public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
				org.apache.spark.mllib.linalg.Vector yi;
				int[] indices = null;
				int i;
				while (arg0.hasNext()) {
					yi = arg0.next();
					indices = ((SparseVector) yi).indices();
					for (i = 0; i < indices.length; i++) {
						internalSumY[indices[i]] += yi.apply(indices[i]);
					}
				}
				matrixAccumY.add(internalSumY);
			}

		});// End Mean Job

		// Get the sum of column Vector from the accumulator and divide each
		// element by the number of rows to get the mean
		// not best of practice to use non-final variable
		final Vector meanVector = new DenseVector(matrixAccumY.value()).divide(nRows);
		final Broadcast<Vector> br_ym_mahout = sc.broadcast(meanVector);

		endTime = System.currentTimeMillis();
		totalTime = endTime - startTime;

		stat.preprocessTime = (double) totalTime / 1000.0;

		stat.totalRunTime = stat.preprocessTime;

		stat.appName = "QRMod1";
		stat.dataSet = dataset;
		stat.nRows = nRows;
		stat.nCols = nCols;

		// compute principal components
		computePrincipalComponents(sc, vectors, br_ym_mahout, meanVector, outputPath, nRows, nCols, nPCs,
				subsample, tolerance, k_plus_one_singular_value, q, maxIterations, calculateError, subsampleNorm);

		// count the average sketch runtime

		for (int j = 0; j < stat.sketchTime.size(); j++) {
			stat.avgSketchTime += stat.sketchTime.get(j);
		}
		stat.avgSketchTime /= stat.sketchTime.size();

		// save statistics
		PCAUtils.printStatToFile(stat, outputPath);

		return null;
	}

	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc,
			JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors, final Broadcast<Vector> br_ym_mahout,
			final Vector meanVector,  String outputPath, final int nRows, final int nCols, final int nPCs,
			final int subsample, final double tolerance, final double k_plus_one_singular_value, final int q,
			final int maxIterations, final int calculateError, final int subsampleNorm) throws FileNotFoundException {

		startTime = System.currentTimeMillis();


		/************************** SSketchPCA PART *****************************/

		/**
		 * Sketch dimension ,S=nPCs+subsample Sketched matrix, B=A*S; QR
		 * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
		 */

		// initialize & broadcast a random seed
		// org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix =
		// org.apache.spark.mllib.linalg.Matrices.randn(nCols,
		// nPCs + subsample, new SecureRandom());
		// //PCAUtils.printMatrixToFile(GaussianRandomMatrix,
		// OutputFormat.DENSE, outputPath+File.separator+"Seed");
		// final Matrix seedMahoutMatrix =
		// PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
		/**
		 * Sketch dimension ,S=nPCs+subsample Sketched matrix, B=A*S; QR
		 * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
		 */

		// initialize & broadcast a random seed
		org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
				nPCs + subsample, new SecureRandom());
		//PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE, outputPath + File.separator + "Seed");
		Matrix B = PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
		// Matrix GaussianRandomMatrix = PCAUtils.randomValidationMatrix(nCols,
		// nPCs + subsample);
		// Matrix B = GaussianRandomMatrix;
		// PCAUtils.printMatrixToFile(PCAUtils.convertMahoutToSparkMatrix(GaussianRandomMatrix),
		// OutputFormat.DENSE, outputPath+File.separator+"Seed");

		final int s=nPCs+subsample;
		// Broadcast Seed because it will be used in several jobs and several
		// iterations.
		Matrix V=null;
		double spectral_error,error,prevError=tolerance+1;
		
		for (int iter = 0; iter < maxIterations&&prevError>tolerance; iter++) {
			
				
			
			final Broadcast<Matrix> seed = sc.broadcast(B);

			Vector seedMu = B.transpose().times(meanVector);
			final Broadcast<Vector> brSeedMu = sc.broadcast(seedMu);
					

			
			JavaRDD<Matrix> Rs = vectors.glom().map(new Function<List<org.apache.spark.mllib.linalg.Vector>, Matrix>() {

				@Override
				public Matrix call(List<org.apache.spark.mllib.linalg.Vector> v1) throws Exception {
					// TODO Auto-generated method stub
					Matrix A = new DenseMatrix(v1.size(), nPCs + subsample);

					for (int i = 0; i < v1.size(); i++) {
						double[] values = v1.get(i).toArray();// TODO check does
																// it
																// really save
																// time?!?!

						int[] indices = ((SparseVector) v1.get(i)).indices();
						int index;
						double value = 0;

						for (int b = 0; b < (nPCs + subsample); b++) {
							for (int a = 0; a < indices.length; a++) {
								index = indices[a];
								value += values[index] * seed.value().getQuick(index, b);
							}
							A.setQuick(i, b, value - brSeedMu.value().getQuick(b));
							value = 0;
						}
					}

					// QR decomposition of B
					int rows = A.rowSize();
					int columns = A.columnSize();

					int min = Math.min(rows, columns);

					Matrix r = new DenseMatrix(min, columns);

					for (int i = 0; i < min; i++) {
						Vector qi = A.viewColumn(i);
						double alpha = qi.norm(2);
						qi.assign(Functions.div(alpha));
						r.set(i, i, alpha);

						for (int j = i + 1; j < columns; j++) {
							Vector qj = A.viewColumn(j);
							double beta = qi.dot(qj);
							r.set(i, j, beta);
							if (j < min) {
								qj.assign(qi, Functions.plusMult(-beta));
							}

						}
					}

					return r;
				}

			});

			Matrix R = Rs.treeReduce(new Function2<Matrix, Matrix, Matrix>() {

				@Override
				public Matrix call(Matrix v1, Matrix v2) throws Exception {
					// TODO Auto-generated method stub
					Matrix v3 = new DenseMatrix(v1.rowSize() + v2.rowSize(), (nPCs + subsample));
					for (int i = 0; i < v1.rowSize(); i++) {
						for (int j = 0; j < v1.columnSize(); j++) {
							v3.setQuick(i, j, v1.getQuick(i, j));
						}
					}
					for (int i = v1.rowSize(); i < v1.rowSize() + v2.rowSize(); i++) {
						for (int j = 0; j < v2.columnSize(); j++) {
							v3.setQuick(i, j, v2.getQuick(i - v1.rowSize(), j));
						}
					}
					org.apache.mahout.math.QRDecomposition QR = new org.apache.mahout.math.QRDecomposition(v3);
					return QR.getR();
				}
			});

			R = PCAUtils.inv(R);
			// omega-V*V'*omega
			Matrix Seed = B.times(R);

			seedMu = Seed.transpose().times(meanVector);
			final Broadcast<Vector> brSeedMu_R = sc.broadcast(seedMu);
			// System.out.println(brSeedMu.value().getQuick(5));

			final Broadcast<Matrix> seed_R = sc.broadcast(Seed);

			final Accumulator<double[]> sumQ = sc.accumulator(new double[nPCs + subsample],
					new VectorAccumulatorParam());
			final Accumulator<double[][]> sumQtA = sc.accumulator(new double[(nPCs + subsample)][nCols],
					new MatrixAccumulatorParam());

			final double[] sumQPartial = new double[nPCs + subsample];
			final double[][] sumQtAPartial = new double[(nPCs + subsample)][nCols];

			final int row = nPCs + subsample;

			vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

				@Override
				public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {

					org.apache.spark.mllib.linalg.Vector Avec = null;
					double[] Q = new double[nPCs + subsample];
					double[] A = null;

					while (arg0.hasNext()) {
						// lol mistake
						Avec = arg0.next();

						A = Avec.toArray();// TODO check does
											// it
											// really save
											// time?!?!

						int[] indices = ((SparseVector) Avec).indices();
						int index;
						double value = 0;
						for (int j = 0; j < (nPCs + subsample); j++) {
							for (int i = 0; i < indices.length; i++) {
								index = indices[i];
								value += A[index] * seed_R.value().getQuick(index, j);
							}
							Q[j] = value - brSeedMu_R.value().getQuick(j);
							value = 0;
						}

						for (int j = 0; j < indices.length; j++) {
							for (int i = 0; i < row; i++) {
								index = indices[j];
								sumQtAPartial[i][index] += Q[i] * A[index];
							}
						}
						for (int i = 0; i < row; i++) {
							sumQPartial[i] += Q[i];
						}

					}

					sumQ.add(sumQPartial);
					sumQtA.add(sumQtAPartial);

				}

			});

			final Matrix sumQtAres = new DenseMatrix(sumQtA.value());
			final Vector sumQres = new DenseVector(sumQ.value());
			final Matrix Qtmu = sumQres.cross(meanVector);
			B = sumQtAres.minus(Qtmu);

			org.apache.mahout.math.SingularValueDecomposition SVD 
			 = new org.apache.mahout.math.SingularValueDecomposition(B);

			B = B.transpose();
			
			V = SVD.getV().viewPart(0, nCols, 0, nPCs);
			
			endTime = System.currentTimeMillis();
			totalTime = endTime - startTime;
			double time= (double) totalTime / 1000.0;
			stat.sketchTime.add(time);
			stat.totalRunTime += time;
			
			if (calculateError == 1) {
				// log.info("Computing the error at round " + round + " ...");
				System.out.println("Computing the error at round " + iter + " ...");
				stat.nIter++;
				// the following subsample is fixed
				spectral_error = norm(sc, vectors, nRows, nCols, 1, subsampleNorm, q, meanVector, V,br_ym_mahout);
				error = (spectral_error - k_plus_one_singular_value) / k_plus_one_singular_value;

				stat.errorList.add((Double) error);
				// log.info("... end of computing the error at round " + round +
				// " And error=" + error);
				System.out.println("... end of computing the error at round " + iter + " error=" + error);
				prevError = error;
			}
			
			//if(dw<=tolerance) break;
			/**
			 * reinitialize
			 */
			startTime = System.currentTimeMillis();
			
			
		
		}

		
		return PCAUtils
				.convertMahoutToSparkMatrix(V);

	}

	private static double norm(JavaSparkContext sc, final JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors,
			final int nRows, final int nCols, final int nPCs, final int subsample, final int q, final Vector meanVector,
			final Matrix centralC, final Broadcast<Vector> br_ym_mahout) {
		/************************** SSketchPCA PART *****************************/

		/**
		 * Sketch dimension ,S=s Sketched matrix, B=A*S; QR decomposition,
		 * Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
		 */

		// initialize & broadcast a random seed
		org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
				nPCs + subsample, new SecureRandom());
		// PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE,
		// outputPath+File.separator+"Seed");
		Matrix B = PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
		
		System.out.println(PCAUtils.convertMahoutToSparkMatrix(centralC));
		Matrix V = new org.apache.mahout.math.SingularValueDecomposition(centralC).getU();

		org.apache.mahout.math.SingularValueDecomposition SVD = null;

		double S = 0;

		for (int iter = 0; iter < q; iter++) {
			// V'*omega
			final Matrix VtSeed = V.transpose().times(B);
			// V*V'*omega
			final Matrix VVtSeed = V.times(VtSeed);
			// omega-V*V'*omega
			Matrix Seed = B.minus(VVtSeed);
			
			// System.out.println(brSeedMu.value().getQuick(5));


			final int s=nPCs+subsample;
			// Broadcast Seed because it will be used in several jobs and several
			// iterations.
			final Broadcast<Matrix> br_Seed = sc.broadcast(Seed);

			// Zm = Ym * Seed
			Vector zm_mahout = new DenseVector(s);
			zm_mahout = PCAUtils.denseVectorTimesMatrix(br_ym_mahout.value(), Seed, zm_mahout);

			// Broadcast Zm because it will be used in several iterations.
			final Broadcast<Vector> br_zm_mahout = sc.broadcast(zm_mahout);
			// We skip computing Z as we generate it on demand using Y and Seed

			// 3. Z'Z and Y'Z Job: The job computes the two matrices Z'Z and Y'Z
			/**
			 * Zc = Yc * MEM (MEM is the in-memory broadcasted matrix seed)
			 * 
			 * ZtZ = Zc' * Zc
			 * 
			 * YtZ = Yc' * Zc
			 * 
			 * It also considers that Y is sparse and receives the mean vectors Ym
			 * and Xm separately.
			 * 
			 * Yc = Y - Ym
			 * 
			 * Zc = Z - Zm
			 * 
			 * Zc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = Z - Zm
			 * 
			 * ZtZ = (Z - Zm)' * (Z - Zm)
			 * 
			 * YtZ = (Y - Ym)' * (Z - Zm)
			 * 
			 */
			final Accumulator<double[][]> matrixAccumZtZ = sc.accumulator(new double[s][s],
					new MatrixAccumulatorParam());
			final Accumulator<double[][]> matrixAccumYtZ = sc.accumulator(new double[nCols][s],
					new MatrixAccumulatorParam());
			final Accumulator<double[]> matrixAccumZ = sc.accumulator(new double[s], new VectorAccumulatorParam());

			/*
			 * Initialize the output matrices and vectors once in order to avoid
			 * generating massive intermediate data in the workers
			 */
			final double[][] resArrayYtZ = new double[nCols][s];
			final double[][] resArrayZtZ = new double[s][s];
			final double[] resArrayZ = new double[s];

			/*
			 * Used to sum the vectors in one partition.
			 */
			final double[][] internalSumYtZ = new double[nCols][s];
			final double[][] internalSumZtZ = new double[s][s];
			final double[] internalSumZ = new double[s];

			vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

				public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
					org.apache.spark.mllib.linalg.Vector yi;
					while (arg0.hasNext()) {
						yi = arg0.next();

						/*
						 * Perform in-memory matrix multiplication zi = yi' * Seed
						 */
						PCAUtils.sparseVectorTimesMatrix(yi, br_Seed.value(), resArrayZ);

						// get only the sparse indices
						int[] indices = ((SparseVector) yi).indices();

						PCAUtils.outerProductWithIndices(yi, br_ym_mahout.value(), resArrayZ, br_zm_mahout.value(),
								resArrayYtZ, indices);
						PCAUtils.outerProductArrayInput(resArrayZ, br_zm_mahout.value(), resArrayZ, br_zm_mahout.value(),
								resArrayZtZ);
						int i, j, rowIndexYtZ;

						// add the sparse indices only
						for (i = 0; i < indices.length; i++) {
							rowIndexYtZ = indices[i];
							for (j = 0; j < s; j++) {
								internalSumYtZ[rowIndexYtZ][j] += resArrayYtZ[rowIndexYtZ][j];
								resArrayYtZ[rowIndexYtZ][j] = 0; // reset it
							}

						}
						for (i = 0; i < s; i++) {
							internalSumZ[i] += resArrayZ[i];
							for (j = 0; j < s; j++) {
								internalSumZtZ[i][j] += resArrayZtZ[i][j];
								resArrayZtZ[i][j] = 0; // reset it
							}

						}
					}
					matrixAccumZ.add(internalSumZ);
					matrixAccumZtZ.add(internalSumZtZ);
					matrixAccumYtZ.add(internalSumYtZ);
				}

			});// end Z'Z and Y'Z Job

			/*
			 * Get the values of the accumulators.
			 */
			Matrix centralYtZ = new DenseMatrix(matrixAccumYtZ.value());
			Matrix centralZtZ = new DenseMatrix(matrixAccumZtZ.value());
			Vector centralSumZ = new DenseVector(matrixAccumZ.value());

			
			centralYtZ = PCAUtils.updateXtXAndYtx(centralYtZ, centralSumZ, br_ym_mahout.value(), zm_mahout, nRows);
			centralZtZ = PCAUtils.updateXtXAndYtx(centralZtZ, centralSumZ, zm_mahout, zm_mahout, nRows);

			
			Matrix R = new org.apache.mahout.math.CholeskyDecomposition(centralZtZ, false).getL().transpose();
			
			
			R = PCAUtils.inv(R);
			centralYtZ=centralYtZ.times(R);
			centralYtZ=centralYtZ.transpose();
			

			final Matrix QtAV = centralYtZ.times(V);
			final Matrix QtAVVt = QtAV.times(V.transpose());
			B = centralYtZ.minus(QtAVVt);

			SVD = new org.apache.mahout.math.SingularValueDecomposition(B);

			B = B.transpose();

			Double newS = SVD.getS().getQuick(nPCs - 1, nPCs - 1);
			newS = Math.round(newS * 10000.0) / 10000.0;
			if (newS == S)
				break;
			else
				S = newS;
			System.out.println(S);
		}

		return S;

	}

	private static void printLogMessage(String argName) {
		log.error("Missing arguments -D" + argName);
		log.info(
				"Usage: -Di=<path/to/input/matrix> -Do=<path/to/outputfolder> -Drows=<number of rows> -Dcols=<number of columns> -Dpcs=<number of principal components> [-DerrSampleRate=<Error sampling rate>] [-DmaxIter=<max iterations>] [-DoutFmt=<output format>] [-DcalculateError=<0/1 (compute projected matrix or not)>]");
	}
}