package utils;

import java.util.ArrayList;

import jmetal.core.Solution;
import jmetal.encodings.variable.Binary;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class WekaUtils {
	
	/**
	 * Retorna um objeto Instances contendo apenas as instâncias selecionadas a partir do AG
	 * @param instances
	 * @param solution
	 * @return
	 */
	public static Instances getSelectedInstances(Instances instances, Solution solution){
		
		//System.out.println("tamanho da base completa: " + instances.size());
		
		Binary sol = (Binary) solution.getDecisionVariables()[0];
		int bits = sol.getNumberOfBits();
		
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		for (int i = 0; i < instances.numAttributes(); i++) {
			attInfo.add(instances.attribute(i));
		}
		
		Instances selectedInstances = new Instances("teste", attInfo, instances.size());
		selectedInstances.setClassIndex(instances.classIndex());
		
		for (int i = 0; i < bits; i++) {
			if (sol.getIth(i)) {
				selectedInstances.add(instances.get(i));
			}
		}
		
		//System.out.println("tamanho da solução: " + selectedInstances.size());
		
		return selectedInstances;
		
	}
	
	/**
	 * Retorna a classe nominal de uma instância
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public static String getInstanceClass(Instance instance) throws Exception{
		if(instance.classIndex() == -1){
			throw new Exception("Classa indefinida");
		}
		//String classeNominalReal = test.classAttribute().value((int) test.instance(i).classValue());
		return instance.stringValue(instance.classIndex());
		
	}
	
	/**
	 * Retorna o nome do atributo que corresponde a classe 
	 * @param instances
	 * @return
	 */
	public static String getClassAttributeName(Instances instances){
		return instances.attribute(instances.classIndex()).name();
	}

}
