# sSketch
The source code for paper: sSketch: A Scalable Sketching Technique for PCA in the Cloud

-------------Cluster Info--------------------------------
1. Go to AWS Console and select EMR
2. Press Create Cluster and choose Advanced options
3. Under the options of EMR choose EMR-5.7.0, and select necessary softwares as mentioned in the paper
4. Click Next, choose the type and number of instance (m3.xlarge, 1 master and 7 slaves in our case)
5. Click Next, choose your .pem file to associate your account
6. Create Cluster

After creating cluster, two additional steps are required to show the web UI for spark, ganglia and yarn
1. In linux console type: ssh -i testMine.pem -N -D 8157 hadoop@master-public-dns 
2. Remove and reinstall foxyproxy in chrome and add the settings file from Amazon
(I have observed that if you have multiple clusters running, only one port forward for one master will
work for all)

Also additionally install 'tmux' : sudo yum install tmux


----------To run MLLib-PCA and Mahout-PCA ----------------------------------

1. First type: mahout spark-shell --driver-memory 10G --executor-memory 10G. If you have a spark-shell up and running, you may want to close it first by :  fuser -k 4040/tcp  

2. Include the following libraries:

	import org.apache.spark.mllib.linalg.distributed.RowMatrix 
	import org.apache.hadoop.io.IntWritable  
	import org.apache.mahout.math.VectorWritable 
	import scala.collection.JavaConverters._  

3. For MLlib-PCA:

	val t1=System.nanoTime
	val rdd = sc.sequenceFile("s3n://YOUR-REPO/FILE.seq", classOf[IntWritable], classOf[VectorWritable],PARTITION). map{case (x, y) => (org.apache.spark.mllib.linalg.Vectors.sparse(y.get.size,y.get.nonZeroes.iterator.asScala.map(e=>(e.index(),e.get)).toSeq))}
	//replace the capital words as necessary
	val mat= new RowMatrix(rdd)
	val pc= mat.computePrincipalComponents(10)  //give your desired number of k 
	val t2=System.nanoTime
	val t=t2-t1
	t
	
4. For Mahout-PCA
	val t1=System.nanoTime
	val result: org.apache.spark.rdd.RDD[(Int, Vector)] = sc.sequenceFile("s3n://YOUR-REPO/FILE.seq", classOf[IntWritable], classOf[VectorWritable],PARTITION). map{case (x, y) => (x.get(), new SequentialAccessSparseVector(y.get()))} 
	val matrix = org.apache.mahout.sparkbindings.drmWrap(result)
	val (_,_,s)=org.apache.mahout.math.decompositions.dspca(matrix, k=10, p=10,q=0) //choose as necessary
	val t2=System.nanoTime
	val t=t2-t1
	t

-------To Run sSketch-PCA----------------------

1. First build the file by: mvn clean package

2. From the target folder, take the jar file and upload somewhere in the cloud, make the link public,

3. ssh into cluster's master (-o ServerAliveInterval=60 recommended) and type: curl -O "URL-to-JAR"

4. Make a directory called output: mkdir output and type: tmux. All the scripts below should be run in tmux console in case, your connection suddenly breaks. To attach a tmux session: tmux attach, to detach: ctrl+b,d, to scroll and see results ctrl+b,[ and then up/page up.

5. Finally choose from below:

	a. To get the Singular Value first, if we want to compute 10 principal components, we will need 11th singular value:

  	spark-submit --master yarn --verbose --class org.ssketch.SingularValue --conf spark.driver.maxResultSize=0 --conf spark.network.timeout=4000s --conf spark.executor.instances=<put #of instances> --conf spark.executor.cores=8 --driver-memory <Highest Memory Available> --executor-memory <Highest Memory Available> --driver-java-options "-Di=s3n://YOUR-REPO/FILE.seq -Do=output/ -Drows=<N> -Dcols=<D> -Dclusters=<PARTITION> -Dq=<#iterations> -Dpcs=<k+1> -DsubSample=10 -DsubSampleNorm=10 -DmaxIter=1 -DcalculateError=0 -DSingularValue=0 -Dtolerance=0.05" sSketch.jar 

  	b. To calculate principal components:

  	spark-submit --master yarn --verbose --class org.ssketch.SSketchPCA --conf spark.driver.maxResultSize=0 --conf spark.network.timeout=4000s --conf spark.executor.instances=<put #of instances> --conf spark.executor.cores=8 --driver-memory <Highest Memory Available> --executor-memory <Highest Memory Available> --driver-java-options "-Di=s3n://YOUR-REPO/FILE.seq -Do=output/ -Drows=<N> -Dcols=<D> -Dclusters=<PARTITION> -Dq=<#iterations> -Dpcs=<k> -DsubSample=10 -DsubSampleNorm=10 -DmaxIter=1 -DcalculateError=0 -DSingularValue=<Derived Singular Value> -Dtolerance=0.05" sSketch.jar 
=======
