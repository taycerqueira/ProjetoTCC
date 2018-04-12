package algoritmoGenetico;

import jmetal.core.Solution;
import jmetal.encodings.variable.Binary;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import sistemaFuzzy.SistemaFuzzy;

public class Fitness {	
	
	public static double[] calcAccuracyAndReductionKnn(Solution solution, Instances instances, Instances trainKnn) throws Exception{
		
		double[] resultado = new double[2];
		//Binary sol = (Binary) solution.getDecisionVariables()[0];
		Binary sol = (Binary) solution.getDecisionVariables()[0];
		int bits = sol.getNumberOfBits();
		int count = 0;
		int acertos = 0;
		
		//System.out.println("bits: " + bits);
		//System.exit(0);
		
		Classifier ibk = new IBk();	
		ibk.buildClassifier(trainKnn);
		
		//Cada instância é uma posição no cromossomo (indice do cromossomo é o índice da instaância)
		for (int i = 0; i < bits; i++) {		
			
			if (sol.getIth(i)) {//Se a instância faz parte da base, testo ela usando o KNN
				
				count++;
				
				Instance instancia = instances.get(i);
				String classeNominalReal = instances.classAttribute().value((int) instances.instance(i).classValue());
				
				double classe = ibk.classifyInstance(instancia);
				String classeNominalKnn = instances.classAttribute().value((int) classe);
				
                if (classeNominalReal.equals(classeNominalKnn)) {
					acertos++;
                }
				
			}

		}
		
		double acuracia =  (double)acertos/(double)count;
		
		resultado[0] = acuracia;
		resultado[1] = (instances.size() - count) / (double) instances.size();
		
		return resultado;
		
	}

	public static double[] calcAccuracyAndReductionFuzzy(Solution solution, Instances instances, Instances trainFuzzy) throws Exception{
		
		double[] resultado = new double[2];

		SistemaFuzzy fuzzy = new SistemaFuzzy(SistemaFuzzy.INFERENCIA_GERAL, "polarity");
		resultado = fuzzy.calcAccuracyAndReductionSolution(trainFuzzy, instances, solution);

		return resultado;
		
	}
	
	public static double calcAccuracyKnn(Instances instances, Instances trainKnn){
		
		int count = 0;
		int acertos = 0;
		
		Classifier ibk = new IBk();	
		try {
			ibk.buildClassifier(trainKnn);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//Cada instância é uma posição no cromossomo (indice do cromossomo é o índice da instaância)
		for (int i = 0; i < instances.numInstances(); i++) {		
			
			count++;
			
			Instance instancia = instances.get(i);
			String classeNominalReal = instances.classAttribute().value((int) instances.instance(i).classValue());
					
			try {
				double classe;
				classe = ibk.classifyInstance(instancia);
				String classeNominalKnn = instances.classAttribute().value((int) classe);
	            if (classeNominalReal.equals(classeNominalKnn)) {
					acertos++;
	            }
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
		
		return (double)acertos/(double)count;

	}
	
	

}
