package algoritmoGenetico;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;
import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import sistemaFuzzy.SistemaFuzzy;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args) throws Exception {
		
//		String databaseName = "basefilmes_53atributos.arff";
//		String classAttribute = "polarity";
		
		String databaseName = "weka-database/iris.arff";
		String classAttribute = "class";
		
		DataSource source = new DataSource (databaseName);
		Instances data = source.getDataSet();
		
		data.setClass(data.attribute(classAttribute));
		
		/* Parâmetros para o AG */
		//int populationSizeSelectInstances          = 10;
		int populationSizeSelectInstances          = 200;
        int maxEvaluationsSelectInstances          = 1000;        
        double probabilityCrossoverSelectInstances = 0.4;
        double probabilityMutationSelectInstances  = 0.4;

		int seed = 1;          // the seed for randomizing the data
		int folds = 10;         // the number of folds to generate, >=2

		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(data);   // create copy of original data
		randData.randomize(rand);         // randomize data with number generator

		randData.stratify(folds);
		
		double[] acuraciasSemOtimizacao = new double[folds];
		double[] acuraciasComOtimizacao = new double[folds];
		
		SistemaFuzzy fuzzy = new SistemaFuzzy(SistemaFuzzy.INFERENCIA_GERAL, classAttribute);
		System.out.println("Base de dados: " + databaseName + "\n");

		for (int n = 0; n < folds; n++) {
			
			System.out.println("============================================ FOLD "+ n +" ============================================");

			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);
			
	    	//Vou separar aqui 25% da base para treinar o KNN
			int seed2 = 2;          // the seed for randomizing the data
			int folds2 = 5;
			Random rand2 = new Random(seed2);   // create seeded number generator
			Instances randData2 = new Instances(train);   // create copy of original data
			randData.randomize(rand2);         // randomize data with number generator
			
			Instances trainAg = randData2.trainCV(folds2, 0);
			Instances trainKnn = randData2.testCV(folds2, 0);
			
			System.out.println("Tamanho da base de treinamento do AG: " + trainAg.size());
			System.out.println("Tamanho da base de treinamento do KNN: " + trainKnn.size());
			System.out.println("Tamanho da base de teste do Fuzzy: " + test.size());
			
			//Executar fuzzy com a base completa e pegar a acurácia
			
			System.out.println("Executando sistema fuzzy com a base de dados completa...");
			double acuracia1 = fuzzy.calcAccuracy(trainAg, test);
			System.out.println("	Acurácia: " + acuracia1);
			acuraciasSemOtimizacao[n] = acuracia1;
			
			System.out.println("Executando KNN com a base de dados completa...");
			double acuraciaKnn = Fitness.calcAccuracyKnn(trainAg, trainKnn);
			System.out.println("	Acurácia KNN: " + acuraciaKnn);
			
			//Executrar o AG e obter uma base otimizada
			//System.out.println("\nExecutando algoritmo genético...");            
			Solution solution = executaAG(trainAg, trainKnn, populationSizeSelectInstances, maxEvaluationsSelectInstances,
				      probabilityCrossoverSelectInstances, probabilityMutationSelectInstances);
			
			//Executar fuzzy novamente com a base otimizada
			
			//System.out.println("solution crowding distance: " + solution.getCrowdingDistance());

			System.out.println("\nExecutando sistema fuzzy com a base de dados otimizada...");
			double[] resultado = fuzzy.calcAccuracyAndReductionSolution(trainAg, test, solution);
			System.out.println("	Acurácia: " + resultado[0]);
			//System.out.println("	Redução: " + resultado[1]);
			acuraciasComOtimizacao[n] = resultado[0];
			
			System.out.println("\n");
			//System.exit(0);

		}

		System.out.println("=> RESULTADOS:");
		System.out.println("* Fuzzy com base completa: acuracia media = " + calculaMedia(acuraciasSemOtimizacao) + " | desvio padrao: " + calculaDesvioPadrao(acuraciasSemOtimizacao));
//		System.out.println("=> Acurácia média SEM otimização: " + calculaMedia(acuraciasSemOtimizacao));
//		System.out.println("=> Desvio padrão SEM otimização: " + calculaDesvioPadrao(acuraciasSemOtimizacao));
		
		System.out.println("* Fuzzy com base otimizada: acuracia media = " + calculaMedia(acuraciasComOtimizacao) + " | desvio padrao: " + calculaDesvioPadrao(acuraciasComOtimizacao));
//		System.out.println("=> Acurácia média COM otimização: " +calculaMedia(acuraciasComOtimizacao));
//		System.out.println("=> Desvio padrão COM otimização: " + calculaDesvioPadrao(acuraciasComOtimizacao));	

	}
	
	private static double calculaDesvioPadrao(double[] valores){
		
		double media = 0;
		
		media = calculaMedia(valores);
		
		double aux1 = 0;
		for (int i = 0; i < valores.length; i++) {
	        double aux2 = valores[i] - media;
	        aux1 += aux2 * aux2;
			
		}
		
		return Math.sqrt(aux1 / (valores.length - 1));
		
	}
	
	private static double calculaMedia(double[] valores){

		double somatorio = 0;
		
		for (int i = 0; i < valores.length; i++) {
			somatorio += valores[i];						
		}
		
		return somatorio/valores.length;
		
	}
	
	
	private static Solution executaAG(Instances trainAg, Instances trainKnn,  int populationSizeSelectInstances, int maxEvaluationsSelectInstances,
		       double probabilityCrossoverSelectInstances, double probabilityMutationSelectInstances) throws Exception{
		
		long inicio = System.currentTimeMillis(); 
		
		Problem problem = new SelectInstances(trainAg, trainKnn);

        //System.out.println("Problem name....................: " + problem.getName());
        //System.out.println ("Number of variables.............: " + problem.getNumberOfVariables());
        //System.out.println ("Type of solution................: " + problem.getSolutionType().getClass());
		//System.out.println("Size of database train: " + train.size());
         
        
        Algorithm algorithm = new NSGAII_SelectInstances(problem);
        algorithm.setInputParameter("populationSize", populationSizeSelectInstances);
        algorithm.setInputParameter("maxEvaluations", maxEvaluationsSelectInstances);        

        /*System.out.println("Population size.................: " + algorithm.getInputParameter("populationSize").toString());
        System.out.println ("Max evaluations.................: " + algorithm.getInputParameter("maxEvaluations").toString());*/
         
        HashMap<String, Double> parameters = new HashMap<String, Double>();
        parameters.put("probability", probabilityCrossoverSelectInstances);
        
        Operator crossover = CrossoverFactory.getCrossoverOperator("HUXCrossover", parameters);

        parameters = new HashMap<String, Double>();
        parameters.put("probability", probabilityMutationSelectInstances);
        Operator mutation = MutationFactory.getMutationOperator("BitFlipMutation", parameters);

        parameters = null;
        Operator selection = SelectionFactory.getSelectionOperator("BinaryTournament2", parameters);

        algorithm.addOperator("crossover", crossover);
        algorithm.addOperator("mutation", mutation);
        algorithm.addOperator("selection", selection);
        
        // Execute the NSGAII AGMO
        SolutionSet population = algorithm.execute();

        Solution finalSolution = ((SelectInstances) problem).getMidPointSolutionFromPareto(population);
        
        double accuracyTra     = finalSolution.getObjective(0);
        double reductionRate   = finalSolution.getObjective(1);
        
		long fim  = System.currentTimeMillis(); 
        
        System.out.println("	Tempo de seleção das instâncias (mm:ss.SSS) = " + new SimpleDateFormat("mm:ss.SSS").format(new Date(fim - inicio)));  
        System.out.println("	Taxa de redução = " + (-1.0 * reductionRate));
        System.out.println("	Acurácia de treinamento (KNN) = " + (-1.0 * accuracyTra));
        
        return finalSolution;

	}

}
