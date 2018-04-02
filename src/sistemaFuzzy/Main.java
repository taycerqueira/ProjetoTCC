package sistemaFuzzy;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import net.sourceforge.jFuzzyLogic.*;
import net.sourceforge.jFuzzyLogic.rule.Rule;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	
	//static String nomeBase = "basefilmes";
	static String nomeBase = "basefilmes_53atributos";
	static int quantConjuntosFuzzy = 3;
	static String classAttribute = "polarity";
	static double porcentagemTeste = 0.3;
	static boolean debug = false;
	
	public static void main(String[] args) throws Exception {
		
		DataSource source = new DataSource (nomeBase + ".arff");
	    Instances data = source.getDataSet();
	    int numInstancias = data.numInstances();
	    
	    data.setClassIndex(data.numAttributes() - 1);
	    
	    System.out.println("* Quantidade de instâncias na base de dados: " + numInstancias);  
	    System.out.println("* Quantidade de atributos: " + data.numAttributes());
	    System.out.println("* Quantidade de conjuntos fuzzy por atributo: " + quantConjuntosFuzzy);
		
	   // int quantidade = (int) (numInstancias*porcentagemTeste);
	    
		//Separo as instâncias que irão ser utilizadas para teste e gravo em uma arquivo de texto
	    //gerarInidicesTeste(quantidade, numInstancias);
	    
	    //Pego as instancias de teste a partir de um arquivo 
		//int[] indicesInstanciasTeste = getIndicesTeste(quantidade);
		//int[] indicesInstanciasTreinamento = getIndicesTreinamento(indicesInstanciasTeste, numInstancias);
		
		//Instances instanciasTeste = getInstanciasTeste(new Instances(data), indicesInstanciasTreinamento);
		//Instances instanciasTreinamento = getInstanciasTreinamento(new Instances(data), indicesInstanciasTeste);
		
		//Testa o sistema utilizando knn
		//testarBaseComKnn(instanciasTeste, instanciasTreinamento);
		
		//Testa o sistema utilizando fuzzy
		//testarBaseComFuzzy(instanciasTeste, instanciasTreinamento, SistemaFuzzy.INFERENCIA_GERAL);
		
		validarFuzzy(data);
		
	}


	
	private static void testarBaseComKnn(Instances instanciasTeste, Instances instanciasTreinamento) throws Exception{
		
		long inicio = System.currentTimeMillis(); 
		
		System.out.println("\n\n=> Testando a base de instâncias com o KNN...");
		System.out.println("=> Tamanho da base de treinamento: " + instanciasTreinamento.numInstances());
		System.out.println("=> Tamanho da base de teste: " + instanciasTeste.numInstances());
		
		double TP = 0; //quantidade de positivos corretamente classificados
		double TN = 0; //quantidade de negativos corretamente classificados
		
		double FP = 0; //quantidade de negativos que foram classificados como positivos
		double FN = 0; //quantidade de positivos que foram classificados como negativos
		
		Classifier ibk = new IBk();	
		
		ibk.buildClassifier(instanciasTreinamento);
		
		for (int i = 0; i < instanciasTeste.numInstances(); i++) {
			
			Instance instancia = instanciasTeste.get(i);
			
			String classeNominalReal = instanciasTeste.classAttribute().value((int) instanciasTeste.instance(i).classValue());

			double classe = ibk.classifyInstance(instancia);
			String classeNominalPrevista = instanciasTeste.classAttribute().value((int) classe);
			
			if(debug){
				
				System.out.println("Classe Nominal Real: " + classeNominalReal);
				System.out.println("Classe Nominal Prevista: " + classeNominalPrevista);
				
			}
			
			if(classeNominalPrevista.equals("positive")){
				
				if(classeNominalReal.equals("positive")){
					TP++;
				}
				else if(classeNominalReal.equals("negative")){
					FP++;
				}
				
			}
			else if(classeNominalPrevista.equals("negative")){
				
				if(classeNominalReal.equals("negative")){
					TN++;
				}
				else if(classeNominalReal.equals("positive")){
					FN++;
				}
			}	

		}
		
		//System.out.println("Quantidade de instâncias testadas: " + indicesTeste.length);
		System.out.println("Acertos: " + (TP + TN));
		System.out.println("Erros: " + (FP + FN));
		
		double acuracia = (TP + TN)/instanciasTeste.numInstances();
		System.out.println("Acurácia: " + acuracia);
		
		double TPR = TP/(TP + FN); // taxa de verdadeiros positivos
		double TNR = TN/(TN + FP); // taxa de verdadeiros negativos
		System.out.println("Taxa de verdadeiros positivos: " + TPR);
		System.out.println("Taxa de verdadeiros negativos: " + TNR);
		
	    long fim  = System.currentTimeMillis();  
	    System.out.println("* Tempo de execução do KNN (min:seg:mil): " + new SimpleDateFormat("mm:ss.SSS").format(new Date(fim - inicio)));
		
	}
	
	private static void validarFuzzy(Instances data) throws Exception{
		
		double[][] resultados = new double[10][3];
		
		 int seed = 1;          // the seed for randomizing the data
		 int folds = 10;         // the number of folds to generate, >=2
		 
		 Random rand = new Random(seed);   // create seeded number generator
		 Instances randData = new Instances(data);   // create copy of original data
		 randData.randomize(rand);         // randomize data with number generator
		 
		 randData.stratify(folds);
		 
		 double somaAcuracias = 0;

		 for (int n = 0; n < folds; n++) {
			 
		   Instances train = randData.trainCV(folds, n);
		   Instances test = randData.testCV(folds, n);
		 
		   resultados[n] = testarBaseComFuzzy(test, train, SistemaFuzzy.INFERENCIA_GERAL);
		   
		   somaAcuracias += resultados[n][0];
		   
		 }
		 
		 double mediaAcuracias = somaAcuracias/10;
		
		double somatorio = 0l;
		
		for (int i = 0; i < 10; i++) {
			double valor = resultados[i][0];
			double result =  valor - mediaAcuracias;
			somatorio = somatorio + (result * result);
		}
		
		double desvioPadrao = Math.sqrt(((double) 1 /( 10-1))* somatorio);
		 
		System.out.println("media acuracias: " + mediaAcuracias);
		System.out.println("desvio padrão: " + desvioPadrao);
		
	}
	
	private static double[] testarBaseComFuzzy(Instances instanciasTeste, Instances instanciasTreinamento, String inferenceType) throws Exception{
		
		long inicio = System.currentTimeMillis(); 
		
		System.out.println("\n\n=> Testando a base de instâncias com Sistema Fuzzy...");
		System.out.println("=> Tamanho da base de treinamento: " + instanciasTreinamento.numInstances());
		System.out.println("=> Tamanho da base de teste: " + instanciasTeste.numInstances());
		
		String functionBlockName = "functionBlock";
		String ruleBlockName = "ruleBlock";
		
		SistemaFuzzy fuzzySystem = new SistemaFuzzy();
		FIS fis = fuzzySystem.generateFis(functionBlockName, ruleBlockName, instanciasTreinamento, 3);
		
		System.out.println("=> Testando Sistema Fuzzy...");
		
		double TP = 0; //quantidade de positivos corretamente classificados
		double TN = 0; //quantidade de negativos corretamente classificados
		
		double FP = 0; //quantidade de negativos que foram classificados como positivos
		double FN = 0; //quantidade de positivos que foram classificados como negativos
		
		for (int k = 0; k < instanciasTeste.numInstances(); k++) {
			
			//System.out.println("Número da instância: " + k);
			Instance instancia = instanciasTeste.get(k);
			
			//System.out.println("Classe real: " + k);
			String classeReal = instanciasTeste.classAttribute().value((int) instanciasTeste.instance(k).classValue());
			
			String classeInferida = null;
			
			//Seta as entradas
			//Considerando que o último atributo é sempre o atributo que corresponde a classe da instância, por isso usa-se o -1
			for (int i = 0; i < (instanciasTeste.numAttributes() - 1); i++) {
				
				if(instanciasTeste.attribute(i).isNumeric()){
	
					String nomeAtributo = instanciasTeste.attribute(i).name();
					double valor = instancia.value(i);
					//System.out.println(nomeAtributo + ": " + valor);
					
					fis.setVariable(nomeAtributo, valor);
				}	
			}
	
			// Evaluate
			fis.evaluate();
			
			if(inferenceType == SistemaFuzzy.INFERENCIA_GERAL){
				
				/************************** FUZZY GERAL *****************************/
				int contPositive = 0;
				double sumPositive = 0;
				int contNegative = 0;
				double sumNegative = 0;
				
				int contRegrasAtivadas = 0;
				
			    for(Rule r : fis.getFunctionBlock(functionBlockName).getFuzzyRuleBlock(ruleBlockName).getRules()){
			    	
			    	double grau = r.getDegreeOfSupport();
			    	String classe = r.getConsequents().getFirst().getTermName();
			    	
			    	if(grau > 0){ //Se a regra foi ativada
			    		
			    		contRegrasAtivadas++;
				    	//System.out.println("indice: "+ k +" | grau: " + grau + " | classe: " + classe);
			    		
				    	if(classe.equals("positive")){
				    		
				    		contPositive++;
				    		sumPositive += grau;
				    		
				    	}
				    	else if(classe.equals("negative")){
				    		
				    		contNegative++;
				    		sumNegative += grau;
				    		
				    	}
				    	
			    	}
			    	
			    }
			    
			    double mediaPositive = sumPositive/contPositive;
			    double mediaNegative = sumNegative/contNegative;
			    
			    //System.out.println("sum negative: " + sumNegative + " | contNegative: " + contNegative);
			    //System.out.println("Quantidade de regras ativadas: " + contRegrasAtivadas);
			    //System.out.println("Media positive: " + mediaPositive);
			    //System.out.println("Media negative: " + mediaNegative);
			    
			    if(mediaPositive >= mediaNegative){
			    	
			    	classeInferida = "positive";
			    	
			    }
			    else{
			    	
			    	classeInferida = "negative";
			    	
			    }  
				
			}
			else if(inferenceType == SistemaFuzzy.INFERENCIA_CLASSICA){
				
				/************************** FUZZY CLÁSSICO *****************************/
				
				if(fis.getVariable(classAttribute).getValue() >= 0){  // -1 -> negative / 1 -> positive
					
					classeInferida = "positive";
					
				}
				else {
					
					classeInferida = "negative";
					
				}
				
				/**********************************************************************/
				
			}
			
			if(classeInferida.equals("positive")){
				
				if(classeReal.equals("positive")){
					TP++;
				}
				else if(classeReal.equals("negative")){
					FP++;
				}
				
			}
			else if(classeInferida.equals("negative")){
				
				if(classeReal.equals("negative")){
					TN++;
				}
				else if(classeReal.equals("positive")){
					FN++;
				}
				
			}

		}
		
		System.out.println("Acertos: " + (TP + TN));
		System.out.println("Erros: " + (FP + FN));
		
		double acuracia = (TP + TN)/instanciasTeste.numInstances();
		System.out.println("Acurácia: " + acuracia);
		
		double TPR = TP/(TP + FN); // taxa de verdadeiros positivos
		double TNR = TN/(TN + FP); // taxa de verdadeiros negativos
		System.out.println("Taxa de verdadeiros positivos: " + TPR);
		System.out.println("Taxa de verdadeiros negativos: " + TNR);
		
		long fim  = System.currentTimeMillis(); 
		System.out.println("* Tempo de execução do Fuzzy (min:seg:mil): " + new SimpleDateFormat("mm:ss.SSS").format(new Date(fim - inicio)));
		
		double[] resultado = new double[3];
		resultado[0] = acuracia;
		resultado[1] = TPR;
		resultado[2] = TNR;
		
		return resultado;
		
	}
	
	private static int[] sortearInstanciasTeste(int quantidade, int numInstancias){
		
		int[] indices = new int[quantidade];
		
		List<Integer> numeros = new ArrayList<Integer>();
		
		for (int i = 0; i < numInstancias; i++) { 
			
		    numeros.add(i);
		    
		}
		
		//Embaralhamos os números:
		Collections.shuffle(numeros);
		
		//Adicionamos os números aleatórios no vetor
		for (int i = 0; i < quantidade; i++) {
			
			indices[i] = numeros.get(i);
			
		}
		
		return indices;
		
	}
	
	//Gera os indices de teste e coloca em um arquivo
	public static void gerarInidicesTeste(int quantidade, int numInstancias){
		
		int[] indicesTeste = sortearInstanciasTeste(quantidade, numInstancias);
		
		File arquivo = new File("indicesTeste.txt");
		
		try(FileWriter fw = new FileWriter(arquivo)){
			for (int indice : indicesTeste) {
				fw.write(indice + "\r\n");
			}
		    fw.flush();
		}catch(IOException ex){
		  ex.printStackTrace();
		}
		
	}
	
	private static boolean isParaTeste(int k, int[] indicesTeste){
		
		boolean resultado = false;
		
		for(int i = 0; i < indicesTeste.length; i++){
			
			if(indicesTeste[i] == k){
				
				resultado = true;
				break;
				
			}
			
		}
		
		return resultado;
	}
	
	//Lê o arquivo e retorna os indices de teste
	public static int[] getIndicesTeste(int quantidade){
		
		int[] indicesTeste = new int[quantidade];
		int i = 0;
		
		File arquivo = new File("indicesTeste.txt");
		try(InputStream in = new FileInputStream(arquivo) ){
		  Scanner scan = new Scanner(in);
		  while(scan.hasNext()){
		    String indice = scan.nextLine();
		    if(indice.length() > 0){
		    	indicesTeste[i] = Integer.parseInt(indice);
		    	i++;
		    }
		  }
		}catch(IOException ex){
		  ex.printStackTrace();
		}
		
		//System.out.println("Quantidade de instâncias de teste = " + indicesTeste.length);
		
		return indicesTeste;
		
	}
	
	public static int[] getIndicesTreinamento(int[] indicesInstanciasTeste, int numInstancias){
		
		int[] indicesTreinamento = new int[numInstancias - indicesInstanciasTeste.length];
		
		int cont = 0;
		
		for(int i = 0; i < numInstancias; i++){
			
			if(!isParaTeste(i, indicesInstanciasTeste)){
				
				indicesTreinamento[cont] = i;
				cont++;
				
			}
			
		}
		
		//System.out.println("Quantidade de instâncias de treinamento = " + indicesTreinamento.length);
		
		return indicesTreinamento;
		
	}
	
	public static Instances getInstanciasTreinamento(Instances instancias, int[] indicesTeste){
		
		ArrayList<Instance> instanciasTeste = new ArrayList<Instance>();
		
		for(int i = 0; i < indicesTeste.length; i++){
			instanciasTeste.add(instancias.get(i));			
		}
		
		instancias.removeAll(instanciasTeste);
			
		//System.out.println("Quantidade de instâncias de treinamento = " + instancias.numInstances());
		
		return instancias;
		
	}
	
	public static Instances getInstanciasTeste(Instances instancias, int[] indicesTreinamento){
		
		ArrayList<Instance> instanciasTreinamento = new ArrayList<Instance>();
		
		for(int i = 0; i < indicesTreinamento.length; i++){
			instanciasTreinamento.add(instancias.get(i));			
		}
		
		instancias.removeAll(instanciasTreinamento);
			
		//System.out.println("Quantidade de instâncias de teste = " + instancias.numInstances());
		
		return instancias;
		
	}
	
	public static void testeKnn(Instances instancias, int[] indicesTeste){
		
		//Instances data = getInstanciasTreinamento(instancias, indicesTeste);
		
		Instances data = instancias;
		
		data.setClassIndex(data.numAttributes() - 1);
		 
		//do not use first and second
		Instance teste1 = data.instance(0);
		Instance teste2 = data.instance(5);
		Instance teste3 = data.instance(10);
		Instance teste4 = data.instance(15);
		
		System.out.println("Classe real da primeira: " + data.classAttribute().value((int) data.instance(0).classValue()));
		System.out.println("Classe real da segunda: " + data.classAttribute().value((int) data.instance(5).classValue()));
		System.out.println("Classe real da terceira: " + data.classAttribute().value((int) data.instance(10).classValue()));
		System.out.println("Classe real da quarta: " + data.classAttribute().value((int) data.instance(15).classValue()));
		
		data.delete(0);
		data.delete(5);
		data.delete(10);
		data.delete(15);
 
		Classifier ibk = new IBk();		
		
		try {
			
			ibk.buildClassifier(data);
			
			double class1 = ibk.classifyInstance(teste1);
			double class2 = ibk.classifyInstance(teste2);
			double class3 = ibk.classifyInstance(teste3);
			double class4 = ibk.classifyInstance(teste4);
	 
			System.out.println("\nClasse prevista da primeira: " + data.classAttribute().value((int) class1));
			System.out.println("Classe prevista da segunda: " + data.classAttribute().value((int) class2));
			System.out.println("Classe prevista da terceira: " + data.classAttribute().value((int) class3));
			System.out.println("Classe prevista da quarta: " + data.classAttribute().value((int) class4));
			
		} catch (Exception e) {
			
			// TODO Auto-generated catch block
			e.printStackTrace();
			
		}
 
	}

}
