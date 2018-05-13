package algoritmoGenetico;

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
import utils.Configuracoes;
import utils.Resultado;
import utils.Utils;
import utils.WekaUtils;
import weka.core.Instances;

public class Main {

	public static void main(String[] args) throws Exception {
				
//		Configuracoes config = new Configuracoes("basefilmes_53atributos.arff", "polarity", 0.5, 0.3, 200, 1000);
		Configuracoes config = new Configuracoes("weka-database/iris.arff", "class", 0.5, 0.3, 16, 1000);
		
		Instances instances = config.getInstances();      

		int seed = 1;          // the seed for randomizing the data
		int folds = 10;         // the number of folds to generate, >=2

		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(instances);   // create copy of original data
		randData.randomize(rand);         // randomize data with number generator

		randData.stratify(folds);
		
//		double[] acuraciasSemOtimizacao = new double[folds];
//		double[] acuraciasComOtimizacao = new double[folds];
		
		Resultado[] resultadosFuzzySemOtimizacao = new Resultado[folds];
		Resultado[] resultadosFuzzyComOtimizacao = new Resultado[folds];
		
		Resultado[] resultadosKnnSemOtimizacao = new Resultado[folds];
		Resultado[] resultadosKnnComOtimizacao = new Resultado[folds];
		
		System.out.println("Base de dados: " + config.getDatabase() + "\n");

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
					
//			System.out.println("Tamanho da base de treinamento do AGMO: " + trainAg.size());
//			System.out.println("Tamanho da base de treinamento para função fitness: " + trainKnn.size());
//			System.out.println("Tamanho da base de teste: " + test.size());
			
			//Executar fuzzy com a base completa e pegar a acurácia
			
			//---------------------- TESTES COM A BASE DE DADOS COMPLETA -----------------------------
			
			System.out.println("\nFUZZY: Executando com a base de dados completa...");
			resultadosFuzzySemOtimizacao[n] = Utils.gerarResultadoFuzzy(trainAg, test);
			
			System.out.println("\nKNN: Executando com a base de dados completa...");
			resultadosKnnSemOtimizacao[n] = Utils.gerarResultadoKnn(trainAg, test);
			
			//----------------------- OTIMIZANDO A BASE DE DADOS COM O AG ----------------------------
			
			//Executrar o AG e obter uma base otimizada
//			System.out.println("\nAG: Executando algoritmo genético...");            
			Solution solution = executaAG(trainAg, trainKnn, config);
			
			//---------------------- TESTES COM A BASE DE DADOS COMPLETA -----------------------------
			
			Instances selectedInstances = WekaUtils.getSelectedInstances(trainAg, solution);
			System.out.println("\nFUZZY: Executando com a base de dados otimizada...");
			resultadosFuzzyComOtimizacao[n] = Utils.gerarResultadoFuzzy(selectedInstances, test);
			resultadosFuzzyComOtimizacao[n].setQtdInstanciasAntes(trainAg.size());
			
			System.out.println("\nKNN: Executando com a base de dados otimizada...");
			resultadosFuzzyComOtimizacao[n] = Utils.gerarResultadoKnn(selectedInstances, test);
			resultadosFuzzyComOtimizacao[n].setQtdInstanciasAntes(trainAg.size());
			
			//-----------------------------------------------------------------------------------------
			
			System.out.println("\n");
			//System.exit(0);

		}

		System.out.println("=> RESULTADOS:");
		printResultados(resultadosFuzzySemOtimizacao, "Fuzzy sem otimização");
		printResultados(resultadosFuzzyComOtimizacao, "Fuzzy com otimização");
		printResultados(resultadosKnnSemOtimizacao, "KNN sem otimização");
		printResultados(resultadosKnnComOtimizacao, "KNN com otimização");
	
	}
	
	public static void printResultados(Resultado[] resultados, String label){
		
		System.out.println("\n-------------------------------------------------------------------------");
		
		System.out.println(label);
		
		double[] acuracias = new double[resultados.length];
		long[] duracoes = new long[resultados.length];
		double[] reducoes = new double[resultados.length];
		double[] qtdInstancias = new double[resultados.length];
		double[] regrasGeradas = new double[resultados.length];
		
		for (int i = 0; i < resultados.length; i++) {
			acuracias[i] = resultados[i].calcularAcuracia();
			duracoes[i] = resultados[i].getDuracao();
			reducoes[i] = resultados[i].calcularReducao();
			qtdInstancias[i] = (double)resultados[i].getQtdInstancias();
			regrasGeradas[i] = (double)resultados[i].getQtdRegras();	
		}
		
		System.out.println("	Acurácia média: " + Utils.calculaMedia(acuracias));
		System.out.println("	Desvio padrão: " + Utils.calculaDesvioPadrao(acuracias));
		System.out.println("	Quantidade média de instâncias: " + Utils.calculaMedia(qtdInstancias));
		System.out.println("	Quantidade média de regras geradas: " + Utils.calculaMedia(regrasGeradas));
		System.out.println("	Tempo médio de execução: " + Utils.calcularTempoMedio(duracoes));
		System.out.println("	Redução média: " + Utils.calculaMedia(reducoes));
		
	}
	
	private static Solution executaAG(Instances trainAg, Instances trainKnn, Configuracoes config) throws Exception{
		
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
        
        double accuracyTra     = finalSolution.getObjective(0);
        double reductionRate   = finalSolution.getObjective(1);
        
		long fim  = System.currentTimeMillis(); 
        
        System.out.println("	Tempo de seleção das instâncias (mm:ss.SSS) = " + Utils.calcularTempoExecucao(inicio, fim));  
//        System.out.println("	Taxa de redução = " + (-1.0 * reductionRate));
//        System.out.println("	Acurácia de treinamento (KNN) = " + (-1.0 * accuracyTra));
        
        return finalSolution;

	}

}
