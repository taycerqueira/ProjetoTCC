package utils;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;

import algoritmoGenetico.NSGAII_SelectInstances;
import algoritmoGenetico.SelectInstances;
import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import sistemaFuzzy.SistemaFuzzy;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Utils {
	
	public static Solution executarAG(Instances trainAg, Instances trainKnn, Configuracoes config) throws Exception{
		
		System.out.println("\nEXECUTANDO SELEÇÃO DE INSTÂNCIAS...");
		
		long inicio = System.currentTimeMillis(); 
		
		Problem problem = new SelectInstances(trainAg, trainKnn);

        //System.out.println("Problem name....................: " + problem.getName());
        //System.out.println ("Number of variables.............: " + problem.getNumberOfVariables());
        //System.out.println ("Type of solution................: " + problem.getSolutionType().getClass());
		//System.out.println("Size of database train: " + train.size());
         
        
        Algorithm algorithm = new NSGAII_SelectInstances(problem);
        algorithm.setInputParameter("populationSize", config.getPopulationSize());
        algorithm.setInputParameter("maxEvaluations", config.getMaxEvaluations());        

        /*System.out.println("Population size.................: " + algorithm.getInputParameter("populationSize").toString());
        System.out.println ("Max evaluations.................: " + algorithm.getInputParameter("maxEvaluations").toString());*/
         
        HashMap<String, Double> parameters = new HashMap<String, Double>();
        parameters.put("probability", config.getProbabilityCrossover());
        
        Operator crossover = CrossoverFactory.getCrossoverOperator("HUXCrossover", parameters);

        parameters = new HashMap<String, Double>();
        double probabilityMutation = 1.0/problem.getNumberOfVariables();
        //parameters.put("probability", config.getProbabilityMutation());
        parameters.put("probability", probabilityMutation);
        Operator mutation = MutationFactory.getMutationOperator("BitFlipMutation", parameters);

        parameters = null;
        Operator selection = SelectionFactory.getSelectionOperator("BinaryTournament2", parameters);

        algorithm.addOperator("crossover", crossover);
        algorithm.addOperator("mutation", mutation);
        algorithm.addOperator("selection", selection);
        
        // Execute the NSGAII AGMO
        SolutionSet population = algorithm.execute();

        Solution finalSolution = ((SelectInstances) problem).getMidPointSolutionFromPareto(population);
        
//        double accuracyTra     = finalSolution.getObjective(0);
//        double reductionRate   = finalSolution.getObjective(1);
        
		long fim  = System.currentTimeMillis(); 
        
//        System.out.println("	Tempo de seleção das instâncias (mm:ss.SSS) = " + Utils.calcularTempoExecucao(inicio, fim));  
//        System.out.println("	Taxa de redução = " + (-1.0 * reductionRate));
//        System.out.println("	Acurácia de treinamento (KNN) = " + (-1.0 * accuracyTra));
        
        return finalSolution;

	}
	
	/**
	 * Retorna um objetivo contendo os resultados da execução do Fuzzy
	 * @param train instâncias de treinamento
	 * @param test instâncias de teste
	 * @return int quantidade de acertos
	 */
	public static Resultado gerarResultadoFuzzy(Instances train, Instances test){
		
		int acertos = 0;

		Resultado res = new Resultado(train.size(), test.size());
		SistemaFuzzy fuzzy = new SistemaFuzzy(SistemaFuzzy.INFERENCIA_GERAL);
		
		try {
			
			res.startTime();
			String[] classesInferidas = fuzzy.run(train, test);
			res.endTime();
			
			res.setQtdRegras(fuzzy.getamanhoBaseRegras());
			
			for (int i = 0; i < test.size(); i++) {
				
				String classeReal = WekaUtils.getInstanceClass(test.get(i));
//				System.out.println("classe real: " + classeReal);
				String classeInferida = classesInferidas[i];
//				System.out.println("classe inferida: " + classeInferida);
				if(classeReal.equals(classeInferida)){
					acertos++;
				}
				
			}
			
			res.setQtdAcertos(acertos);
			
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		return res;
	}
	
	/**
	 * Retorna um objetivo contendo os resultados da execução do KNN
	 * @param train instâncias de treinamento
	 * @param test instâncias de teste
	 * @return int quantidade de acertos
	 */
	public static Resultado gerarResultadoKnn(Instances train, Instances test){
		
		int acertos = 0;
		
		Resultado res = new Resultado(train.size(), test.size());
		int testSize = test.size();
		
		Classifier ibk = new IBk();	
		
		try {
			
			res.startTime();
			ibk.buildClassifier(train);
			res.endTime();
			
			for (int i = 0; i < testSize; i++) {		
				
				Instance instancia = test.get(i);
				String classeNominalReal = WekaUtils.getInstanceClass(instancia);
				
				double classe = ibk.classifyInstance(instancia);
				String classeNominalKnn = test.classAttribute().value((int) classe);
				
	            if (classeNominalReal.equals(classeNominalKnn)) {
					acertos++;
	            }

			}
			
			res.setQtdAcertos(acertos);
			
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		return res;
	}
	
	public static String calcularTempoExecucao(long inicio, long fim){
		return new SimpleDateFormat("mm:ss.SSS").format(new Date(fim - inicio));
	}
	
	public static double calculaDesvioPadrao(double[] valores){
		
		double media = 0;
		
		media = calculaMedia(valores);
		
		double aux1 = 0;
		for (int i = 0; i < valores.length; i++) {
	        double aux2 = valores[i] - media;
	        aux1 += aux2 * aux2;
			
		}
		
		return Math.sqrt(aux1 / (valores.length - 1));
		
	}
	
	public static double calculaMedia(double[] valores){

		double somatorio = 0;
		
		for (int i = 0; i < valores.length; i++) {
			somatorio += valores[i];						
		}
		
		return somatorio/valores.length;
		
	}
	
	public static String calcularTempoMedio(long[] duracoes){
		
		long total = 0;
		for (long l : duracoes) {
			total += l;
		}
		return new SimpleDateFormat("mm:ss.SSS").format(new Date(total));
		
	}

}
