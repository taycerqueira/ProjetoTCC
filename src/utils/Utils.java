package utils;

import java.text.SimpleDateFormat;
import java.util.Date;

import sistemaFuzzy.SistemaFuzzy;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Utils {
	
	/**
	 * Retorna um objetivo contendo os resultados da execução do Fuzzy
	 * @param train instâncias de treinamento
	 * @param test instâncias de teste
	 * @return int quantidade de acertos
	 */
	public static Resultado gerarResultadoFuzzy(Instances train, Instances test){
		
		int acertos = 0;

		Resultado res = new Resultado();
		res.setQtdInstancias(train.size());
		SistemaFuzzy fuzzy = new SistemaFuzzy(SistemaFuzzy.INFERENCIA_GERAL);
		
		try {
			
			res.startTime();
			String[] classesInferidas = fuzzy.run(train, test);
			res.endTime();
			
			res.setQtdRegras(fuzzy.getamanhoBaseRegras());
			
			for (int i = 0; i < test.size(); i++) {
				
				String classeReal = WekaUtils.getInstanceClass(test.get(i));
				String classeInferida = classesInferidas[i];
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
		
		Resultado res = new Resultado();
		res.setQtdInstancias(train.size());
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
