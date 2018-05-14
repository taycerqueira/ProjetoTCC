package utils;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Configuracoes {
	
	public String database;
	public String classAttributeName;
	public double probabilityCrossover;
	public double probabilityMutation;
	public int populationSize;
	public int maxEvaluations;
	private Instances instances;
	
	public Configuracoes(String database, String classAttributeName, double pCrossover,
			double pMutation, int populationSize, int maxEvaluations) throws Exception {
		this.database = database;
		this.classAttributeName = classAttributeName;
		this.probabilityCrossover = pCrossover;
		this.probabilityMutation = pMutation;
		this.populationSize = populationSize;
		this.maxEvaluations = maxEvaluations;
		
		initInstances();
	}
	
	public Configuracoes(String database, String classAttributeName) throws Exception {
		
		this.database = database;
		this.classAttributeName = classAttributeName;
		
		initInstances();
		
		this.populationSize = (instances.size()/100) * 10; //10% do tamanho total
		this.maxEvaluations = 1000;
		this.probabilityMutation = 0.5; //0.5%-1%
		this.probabilityCrossover = 0.6; //80%-95%
		
	}
	
	private void initInstances() throws Exception{
		DataSource source = new DataSource(this.database);
		instances = source.getDataSet();
		instances.setClass(instances.attribute(classAttributeName));
	}
	
	public String getDatabase() {
		return database;
	}

	public void setDatabase(String database) {
		this.database = database;
	}

	public String getClassAttributeName() {
		return classAttributeName;
	}

	public void setClassAttributeName(String classAttributeName) {
		this.classAttributeName = classAttributeName;
	}

	public double getProbabilityCrossover() {
		return probabilityCrossover;
	}

	public void setProbabilityCrossover(double probabilityCrossover) {
		this.probabilityCrossover = probabilityCrossover;
	}

	public double getProbabilityMutation() {
		return probabilityMutation;
	}

	public void setProbabilityMutation(double probabilityMutation) {
		this.probabilityMutation = probabilityMutation;
	}

	public int getPopulationSize() {
		return populationSize;
	}
	
	public void setPopulationSize(int populationSize) {
		this.populationSize = populationSize;
	}

	public int getMaxEvaluations() {
		return maxEvaluations;
	}

	public void setMaxEvaluations(int maxEvaluations) {
		this.maxEvaluations = maxEvaluations;
	}

	public Instances getInstances(){	
		return instances;
	}
	
}
