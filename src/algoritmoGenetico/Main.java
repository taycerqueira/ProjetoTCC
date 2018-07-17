package algoritmoGenetico;

import java.util.Random;
import jmetal.core.Solution;
import utils.Configuracoes;
import utils.Resultado;
import utils.Utils;
import utils.WekaUtils;
import weka.core.Instances;

public class Main {

	public static void main(String[] args) throws Exception {
				
//		Configuracoes config = new Configuracoes("polarity-dataset.arff", "polarity");
		Configuracoes config = new Configuracoes("weka-database/iris.arff", "class");
		
		Instances instances = config.getInstances();      

		int seed = 1;          // the seed for randomizing the data
		int folds = 10;         // the number of folds to generate, >=2

		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(instances);   // create copy of original data
		randData.randomize(rand);         // randomize data with number generator

		randData.stratify(folds);
		
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
			
			System.out.println("\nCLASSIFICAÇÃO FUZZY: Executando com a base de dados completa...");
			
			Resultado res1 = Utils.gerarResultadoFuzzy(trainAg, test);
			resultadosFuzzySemOtimizacao[n] = res1;
//			System.out.println("	Acertos: " + res1.getQtdAcertos());
			System.out.println("	Acurácia: " + res1.calcularAcuracia());
			
//			System.out.println("\nKNN: Executando com a base de dados completa...");
//			Resultado res2 = Utils.gerarResultadoKnn(trainAg, test);
//			resultadosKnnSemOtimizacao[n] = res2;
//			System.out.println("	Acertos: " + res2.getQtdAcertos());
//			System.out.println("	Acurácia: " + res2.calcularAcuracia());
			
			//----------------------- SELEÇÃO DE INSTÂNCIAS ----------------------------
			
			//Executrar o AG e obter uma base otimizada
//			System.out.println("\nAG: Executando algoritmo genético...");            
			Solution solution = Utils.executarAG(trainAg, trainKnn, config);
			
			//---------------------- TESTES COM A BASE DE DADOS REDUZIDA -----------------------------
			
			Instances selectedInstances = WekaUtils.getSelectedInstances(trainAg, solution);
			System.out.println("\nCLASSIFICAÇÃO FUZZY: Executando com a base de dados reduzida...");
			Resultado res3 = Utils.gerarResultadoFuzzy(selectedInstances, test);
			resultadosFuzzyComOtimizacao[n] = res3;
			res3.setQtdInstanciasAntes(trainAg.size());
//			System.out.println("	Acertos: " + res3.getQtdAcertos());
			System.out.println("	Acurácia: " + res3.calcularAcuracia());
			
//			System.out.println("\nKNN: Executando com a base de dados otimizada...");
//			Resultado res4 = Utils.gerarResultadoKnn(selectedInstances, test);
//			res4.setQtdInstanciasAntes(trainAg.size());
//			resultadosKnnComOtimizacao[n] = res4;
//			System.out.println("	Acertos: " + res4.getQtdAcertos());
//			System.out.println("	Acurácia: " + res4.calcularAcuracia());
			
			//-----------------------------------------------------------------------------------------
			
			System.out.println("\n");

		}

		System.out.println("=> RESULTADOS:");
		System.out.println(" ---------------------- CLASSIFICAÇÃO FUZZY ----------------------");
		System.out.println("	-> BASE DE DADOS ORIGINAL");
		printResultados(resultadosFuzzySemOtimizacao);
		System.out.println("	-> BASE DE DADOS OTIMIZADA");
		printResultados(resultadosFuzzyComOtimizacao);
//		System.out.println(" ---------------------- KNN ----------------------");
//		System.out.println("	-> ANTES");
//		printResultados(resultadosKnnSemOtimizacao);
//		System.out.println("	-> DEPOIS");
//		printResultados(resultadosKnnComOtimizacao);
	
	}
	
	public static void printResultados(Resultado[] resultados){
		
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
		
		System.out.println("		Acurácia média: " + Utils.calculaMedia(acuracias));
		System.out.println("		Desvio padrão: " + Utils.calculaDesvioPadrao(acuracias));
		System.out.println("		Quantidade média de instâncias: " + Utils.calculaMedia(qtdInstancias));
		System.out.println("		Quantidade média de regras geradas: " + Utils.calculaMedia(regrasGeradas));
		System.out.println("		Tempo médio de execução: " + Utils.calcularTempoMedio(duracoes));
		System.out.println("		Redução média: " + Utils.calculaMedia(reducoes));
		
	}

}
