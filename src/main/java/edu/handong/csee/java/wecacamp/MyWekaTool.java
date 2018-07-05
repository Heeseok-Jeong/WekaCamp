package edu.handong.csee.java.wecacamp;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class MyWekaTool {

	public static void main(String[] args) {
		MyWekaTool myTool = new MyWekaTool();
		
		myTool.doJob(args);
	}
	
	private void doJob(String[] args) {
//		1. 파일 읽어오기
//		2. 전처리 과정
//		3. 트레이닝 해서 모델 만들기 
//		4. 알고리즘 적용 
		 
		// (1) Read training test files
		String trainingArffPath = args[0];
		String testArffPath = args[1];
		try {
			BufferedReader reader = new BufferedReader(new FileReader(trainingArffPath));
			Instances trainingData = new  Instances(reader);
			trainingData.setClassIndex(trainingData.numAttributes()-1);
			reader.close();
			
			reader = new BufferedReader(new FileReader(testArffPath));
			Instances testData = new Instances(reader);
			testData.setClassIndex(trainingData.numAttributes()-1);
			
			
		// (2) Preprocess : Feature selection
			AttributeSelection attrSelector = getAttributesSelectionFilterByGainRatioAttributeEval(trainingData);
			trainingData = selectFeaturesByAttributeSelection(attrSelector, trainingData);
			testData = selectFeaturesByAttributeSelection(attrSelector, testData);
			
		// (3) Build prediction models
			Classifier myModel = new Logistic();//J48();
			myModel.buildClassifier(trainingData);
			
		// (4) Apply my model on test data
			Evaluation eval = new Evaluation(trainingData);
			eval.evaluateModel(myModel, testData);
			
		// (5) Show prediction results
			int i=0;
			for(Prediction prediction:eval.predictions()) {
				String predictedValue = getClassValue(trainingData,prediction.predicted());
				System.out.println("Instance " + (++i) + " " + predictedValue);
			}

			showSummary(eval,testData);
			
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public AttributeSelection getAttributesSelectionFilterByGainRatioAttributeEval(Instances data){

		AttributeSelection filter = new AttributeSelection();  // package weka.filters.supervised.attribute!
		GainRatioAttributeEval eval = new GainRatioAttributeEval();
		Ranker search = new Ranker();
		search.setThreshold(-1.7976931348623157E308);
		search.setNumToSelect(-1);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		try {
			filter.setInputFormat(data);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return filter;
	}
	
	public Instances selectFeaturesByAttributeSelection(AttributeSelection filter,Instances data) {
		Instances newData = null;
		
		try {
			//filter.setInputFormat(data);

			// generate new data
			newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newData;
	}

	String getClassValue(Instances instances, double index) {
//		instances.attribute(11).value(0);
		return instances.attribute(instances.classIndex()).value((int) index);
	}
	
	private void showSummary(Evaluation eval,Instances instances) {
		for(int i=0; i<instances.classAttribute().numValues();i++) {
			System.out.println("\n*** Summary of Class " + instances.classAttribute().value(i));
			System.out.println("Precision " + eval.precision(i));
			System.out.println("Recall " + eval.recall(i));
			System.out.println("F-Measure " + eval.fMeasure(i));
		}
	}

}
