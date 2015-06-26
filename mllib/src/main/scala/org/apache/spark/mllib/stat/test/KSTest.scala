/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.stat.test

import org.apache.commons.math3.distribution.{NormalDistribution, RealDistribution}
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest

import org.apache.spark.rdd.RDD


/**
 * Conduct the two-sided Kolmogorov Smirnov test for data sampled from a
 * continuous distribution. By comparing the largest difference between the empirical cumulative
 * distribution of the sample data and the theoretical distribution we can provide a test for the
 * the null hypothesis that the sample data comes from that theoretical distribution.
 * For more information on KS Test: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
 */
private[stat] object KSTest {

  // Null hypothesis for the type of KS test to be included in the result.
  object NullHypothesis extends Enumeration {
    type NullHypothesis = Value
    val oneSampleTwoSided = Value("Sample follows theoretical distribution.")
    val twoSampleTwoSided = Value("Both samples follow the same distribution.")
  }

  /**
   * Runs a KS test for 1 set of sample data, comparing it to a theoretical distribution
   * @param data `RDD[Double]` data on which to run test
   * @param cdf `Double => Double` function to calculate the theoretical CDF
   * @return KSTestResult summarizing the test results (pval, statistic, and null hypothesis)
   */
  def testOneSample(data: RDD[Double], cdf: Double => Double): KSTestResult = {
    val n = data.count().toDouble
    val localData = data.sortBy(x => x).mapPartitions {
      part =>
        val partDiffs = oneSampleDifferences(part, n, cdf) // local distances
        searchOneSampleCandidates(partDiffs) // candidates: local extrema
        }.collect()
    val ksStat = searchOneSampleStatistic(localData, n) // result: global extreme
    evalOneSampleP(ksStat, n.toInt)
  }

  /**
   * Runs a KS test for 1 set of sample data, comparing it to a theoretical distribution
   * @param data `RDD[Double]` data on which to run test
   * @param createDist `Unit => RealDistribution` function to create a theoretical distribution
   * @return KSTestResult summarizing the test results (pval, statistic, and null hypothesis)
   */
  def testOneSample(data: RDD[Double], createDist: () => RealDistribution): KSTestResult = {
    val n = data.count().toDouble
    val localData = data.sortBy(x => x).mapPartitions {
      part =>
        val partDiffs = oneSampleDifferences(part, n, createDist) // local distances
        searchOneSampleCandidates(partDiffs) // candidates: local extrema
    }.collect()
    val ksStat = searchOneSampleStatistic(localData, n) // result: global extreme
    evalOneSampleP(ksStat, n.toInt)
  }

  /**
   * Calculate unadjusted distances between the empirical CDF and the theoretical CDF in a
   * partition
   * @param partData `Iterator[Double]` 1 partition of a sorted RDD
   * @param n `Double` the total size of the RDD
   * @param cdf `Double => Double` a function the calculates the theoretical CDF of a value
   * @return `Iterator[Double] `Unadjusted (ie. off by a constant) differences between
   *        ECDF (empirical cumulative distribution function) and CDF. We subtract in such a way
   *        that when adjusted by the appropriate constant, the difference will be equivalent
   *        to the KS statistic calculation described in
   *        http://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm
   *        where the difference is not exactly symmetric
   */
  private def oneSampleDifferences(partData: Iterator[Double], n: Double, cdf: Double => Double)
    : Iterator[Double] = {
    // zip data with index (within that partition)
    // calculate local (unadjusted) ECDF and subtract CDF
    partData.zipWithIndex.map {
      case (v, ix) =>
        // dp and dl are later adjusted by constant, when global info is available
        val dp = (ix + 1) / n
        val dl = ix / n
        val cdfVal = cdf(v)
        // if dp > cdfVal the adjusted dp is still above cdfVal, if dp < cdfVal
        // we want negative distance so that constant adjusted gives correct distance
        if (dp > cdfVal) dp - cdfVal else dl - cdfVal
    }
  }

  private def oneSampleDifferences(partData: Iterator[Double], n: Double,
      createDist: () => RealDistribution)
    : Iterator[Double] = {
    val dist = createDist()
    oneSampleDifferences(partData, n, x => dist.cumulativeProbability(x))
  }

  /**
   * Search the unadjusted differences between ECDF and CDF in a partition and return the
   * two extrema (furthest below and furthest above CDF), along with a count of elements in that
   * partition
   * @param partDiffs `Iterator[Double]` the unadjusted differences between ECDF and CDF in a
   *                 partition
   * @return `Iterator[(Double, Double, Double)]` the local extrema and a count of elements
   */
  private def searchOneSampleCandidates(partDiffs: Iterator[Double])
    : Iterator[(Double, Double, Double)] = {
    val initAcc = (Double.MaxValue, Double.MinValue, 0.0)
    val partResults = partDiffs.foldLeft(initAcc) {
      case ((pMin, pMax, pCt), currDiff) =>
        (Math.min(pMin, currDiff), Math.max(pMax, currDiff), pCt + 1)
    }
    Array(partResults).iterator
  }

  /**
   * Find the global maximum distance between ECDF and CDF (i.e. the KS Statistic) after adjusting
   * local extrema estimates from individual partitions with the amount of elements in preceding
   * partitions
   * @param localData `Array[(Double, Double, Double)]` A local array containing the collected
   *                 results of `searchOneSampleCandidates` across all partitions
   * @param n `Double`The size of the RDD
   * @return The one-sample Kolmogorov Smirnov Statistic
   */
  private def searchOneSampleStatistic(localData: Array[(Double, Double, Double)], n: Double)
    : Double = {
    val initAcc = (Double.MinValue, 0.0)
    // adjust differences based on the # of elements preceding it, which should provide
    // the correct distance between ECDF and CDF
    val results = localData.foldLeft(initAcc) {
      case ((prevMax, prevCt), (minCand, maxCand, ct)) =>
        val adjConst = prevCt / n
        val dist1 = Math.abs(minCand + adjConst)
        val dist2 = Math.abs(maxCand + adjConst)
        val maxVal = Array(prevMax, dist1, dist2).max
        (maxVal, prevCt + ct)
    }
    results._1
  }

  /**
   * A convenience function that allows running the KS test for 1 set of sample data against
   * a named distribution
   * @param data the sample data that we wish to evaluate
   * @param distName the name of the theoretical distribution
   * @return KSTestResult summarizing the test results (pval, statistic, and null hypothesis)
   */
  def testOneSample(data: RDD[Double], distName: String): KSTestResult = {
    val distanceCalc =
      distName match {
        case "stdnorm" => () => new NormalDistribution(0, 1)
        case  _ => throw new UnsupportedOperationException(s"$distName not yet supported through" +
          s"convenience method. Current options are:[stdnorm].")
      }

    testOneSample(data, distanceCalc)
  }

  private def evalOneSampleP(ksStat: Double, n: Long): KSTestResult = {
    val pval = 1 - new KolmogorovSmirnovTest().cdf(ksStat, n.toInt)
    new KSTestResult(pval, ksStat, NullHypothesis.oneSampleTwoSided.toString)
  }

  // Two sample methods
  private def evalTwoSampleP(ksStat: Double, n: Long, m: Long): KSTestResult = {
    val pval = new KolmogorovSmirnovTest().approximateP(ksStat, n.toInt, m.toInt)
    new KSTestResult(pval, ksStat, NullHypothesis.twoSampleTwoSided.toString)
  }

  def testTwoSamples(data1: RDD[Double], data2: RDD[Double]): Double = {
    val n1 = data1.count().toDouble
    val n2 = data2.count().toDouble
    val isSample1 = true // we need a way to identify them once co-sorted
    val joinedData = data1.map(x => (x, isSample1)) ++ data2.map(x => (x, !isSample1))
    val localData = joinedData.sortBy(x => x).mapPartitions {
      part => searchTwoSampleCandidates(part, n1, n2)
      }.collect()
    val ksStat = searchTwoSampleStatistic(localData, n1 * n2) // result: global extreme
    ksStat
    // evalTwoSampleP(ksStat, n1.toLong, n2.toLong)
  }

  // not sure if we want to break this up into 2 functinos how we did with 1 sample??
  // we need to keep track of more info here...so the 2 tasks might be best combined
  private def searchTwoSampleCandidates(partData: Iterator[(Double, Boolean)],
      n1: Double,
      n2: Double)
    : Iterator[(Double, Double, Double)] = {
    val initAcc = (Double.MaxValue, // local minimum
        Double.MinValue, // local maximum
        -1.0, // index for first sample
        -1.0, // index for second sample
        0.0, // count of first sample
        0.0) // count of second sample
    val localResults = partData.foldLeft(initAcc) {
      case ((pMin, pMax, ix1, ix2, ct1, ct2), (v, isSample1)) =>
        if (isSample1) {
          val cdf1 = (ix1 + 1) / n1
          val cdf2 = Math.max(ix2, 0) / n2
          val dist = cdf1 - cdf2
          (Math.min(pMin, dist), Math.max(pMax, dist), ix1 + 1, ix2, ct1 + 1, ct2)
        } else {
          val cdf1 = Math.max(ix1, 0) / n1
          val cdf2 = (ix2 + 1) / n2
          val dist = cdf1 - cdf2
          (Math.min(pMin, dist), Math.max(pMax, dist), ix1, ix2 + 1, ct1, ct2 + 1)
        }
    }
    Array((localResults._1, localResults._2, localResults._3 * n2 - localResults._4 * n1)).iterator
  }

  private def searchTwoSampleStatistic(localData: Array[(Double, Double, Double)], n: Double) = {
    val initAcc = (Double.MinValue, 0.0)
    // adjust differences based on the # of elements preceding it, which should provide
    // the correct distance between the 2 ECDFs
    val results = localData.foldLeft(initAcc) {
      case ((prevMax, prevCt), (minCand, maxCand, ct)) =>
        val adjConst = prevCt / n
        val dist1 = Math.abs(minCand + adjConst)
        val dist2 = Math.abs(maxCand + adjConst)
        val maxVal = Array(prevMax, dist1, dist2).max
        (maxVal, prevCt + ct)
    }
    results._1
  }



}

