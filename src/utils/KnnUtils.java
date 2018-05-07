package utils;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KnnUtils{
	
	/**
	 * Calcula a quantidade de acertos do KNN
	 * @param train instâncias de treinamento
	 * @param test instâncias de teste
	 * @return int quantidade de acertos
	 */
	public static int calcularAcertos(Instances train, Instances test){
		
		int acertos = 0;
		int testSize = test.size();
		
		Classifier ibk = new IBk();	
		
		try {
			
			ibk.buildClassifier(train);
			
			for (int i = 0; i < testSize; i++) {		
				
				Instance instancia = test.get(i);
				String classeNominalReal = WekaUtils.getInstanceClass(instancia);
				
				double classe = ibk.classifyInstance(instancia);
				String classeNominalKnn = test.classAttribute().value((int) classe);
				
	            if (classeNominalReal.equals(classeNominalKnn)) {
					acertos++;
	            }

			}
			
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		return acertos;
	}
	
	/**
	 * Retorna um array de tamanho 2 contendo a acurácia utilizando KNN e a taxa de redução da base de teste em relação ao tamanho original
	 * @param train base utilizada na geração da base de regras do Fuzzy
	 * @param test base de teste do Fuzzy
	 * @param originalSize tamanho da base de dados original que será comparada com a base de teste. Caso o valor seja zero, o tamanho da base de teste será comparado com o tamanho da base de treinamento
	 * @return double[] 0 -> acurácia, 1 -> redução
	 */
	public static double[] calcularAcuraciaReducao(Instances train, Instances test, int originalSize){
		
		if(originalSize == 0){
			originalSize = train.size();
		}
		
		int acertos = calcularAcertos(train, test);

		return getArrayAcuraciaReducao(acertos, test.size(), originalSize);
	}
	
	/**
	 * Calcula a acurácia do algoritmo
	 * @param train base utilizada na geração da base de regras do Fuzzy
	 * @param test base de teste do Fuzzy
	 * @return double Acurácia
	 */
	public static double calcularAcuracia(Instances train, Instances test){
		return calcularAcuracia(calcularAcertos(train, test), test.size());
	}
	
	protected static double[] getArrayAcuraciaReducao(int quantAcertos, int quantIntancias, int quantOriginal){
		double[] resultado = new double[2];
		resultado[0] = calcularAcuracia(quantAcertos, quantIntancias);
		resultado[1] = calcularReducao(quantOriginal, quantIntancias);
		return resultado;
	}
	
	protected static double calcularAcuracia(int quantAcertos, int quantIntancias){
		return (double)quantAcertos/(double)quantIntancias;
	}
	
	protected static double calcularReducao(int quantOriginal, int quantAtual){
		return (quantOriginal - quantAtual) / (double) quantOriginal;
	}

}
