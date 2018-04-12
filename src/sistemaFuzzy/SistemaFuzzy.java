package sistemaFuzzy;

import java.util.ArrayList;
import java.util.HashMap;
import jmetal.core.Solution;
import jmetal.encodings.variable.Binary;
import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.defuzzifier.DefuzzifierCenterOfGravitySingletons;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunctionSingleton;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunctionTriangular;
import net.sourceforge.jFuzzyLogic.membership.Value;
import net.sourceforge.jFuzzyLogic.rule.LinguisticTerm;
import net.sourceforge.jFuzzyLogic.rule.Rule;
import net.sourceforge.jFuzzyLogic.rule.RuleBlock;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;
import weka.core.converters.ConverterUtils.DataSource;

public class SistemaFuzzy {
	
	private static int quantConjuntosFuzzy = 3;
	private FIS fis;
	private String inferenceType;
	private String classAttribute;
	
	public static final java.lang.String INFERENCIA_GERAL = "geral";
	public static final java.lang.String INFERENCIA_CLASSICA = "classica";
	
	public static final java.lang.String FUNCTION_BLOCK_NAME = "functionBlock";
	public static final java.lang.String RULE_BLOCK_NAME = "ruleBlock";
	
	public SistemaFuzzy(String inferenceType, String classAttribute){
		
		this.inferenceType = inferenceType;
		this.classAttribute = classAttribute;
		fis = new FIS();
		
	}
	
	public double calcAccuracy(Instances train, Instances test) throws Exception{
		
		WangMendel wm = new WangMendel(train, classAttribute);
		FunctionBlock fb = generateFunctionBlock(train, wm);
		
		return execute(test, fb);
		
	}
	
	
	/**
	 * Este método treina gera a base de regras com wang-mendel utilizando as instâncias presentes na solution
	 */
	public double calcAccuracySolution(Instances train, Instances test, Solution solution) throws Exception{
		
		WangMendel wm = new WangMendel(train, classAttribute, solution);
		FunctionBlock fb = generateFunctionBlock(train, wm);
		
		return execute(test, fb);
		
	}
	
	/**
	 * Este método treina gera a base de regras com wang-mendel utilizando as instâncias presentes na solution
	 */
	public double[] calcAccuracyAndReductionSolution(Instances train, Instances test, Solution solution) throws Exception{
		
		double[] resultado = new double[2];
		
		WangMendel wm = new WangMendel(train, classAttribute, solution);
		FunctionBlock fb = generateFunctionBlock(train, wm);
		
		resultado[0] = execute(test, fb);
		resultado[1] = (train.size() - wm.getQuantInstancias()) / (double) train.size();
		
		return resultado;
		
	}
	
	/*public double calcAccuracySolution(Instances train, Instances test, Solution solution) throws Exception{
		
		Binary sol = (Binary) solution.getDecisionVariables()[0];
		int bits = sol.getNumberOfBits();

		System.out.println("Testing Fuzzy System...");
		//System.out.println("=> Tamanho da base de teste: " + instancias.numInstances());
		
		Instances trainSolution = new Instances(train);
		
		System.out.println("quantidade de bits: " + bits);
		System.out.println("tamanho da base de treinamento: " + trainSolution.size());
		//System.exit(0);
		
		//System.out.println("tamanho da base de otimizada: " + trainOtimizada.size());
		
		//Cada instância � uma posi��o no cromossomo (indice do cromossomo � o �ndice da instância)
		for (int i = 0; i < bits; i++) {		
			
			if (sol.getIth(i) == false) {//Se a instância n�o faz parte da base otimizada, removo ela
				
				//trainSolution.remove(i);
				trainSolution.remove(i);
				
			}

		}
		
		System.out.println("tamanho da base otimizada: " + trainSolution.size());
		System.exit(0);
		
		double acuracia = execute(trainSolution, test);
		
		return acuracia;
		
	}*/
	
	//Retorna a acurácia
	private double execute(Instances instancias, FunctionBlock fb){
		
		fis.addFunctionBlock(SistemaFuzzy.FUNCTION_BLOCK_NAME, fb);
		
		double TP = 0; //quantidade de positivos corretamente classificados
		double TN = 0; //quantidade de negativos corretamente classificados
		
		double FP = 0; //quantidade de negativos que foram classificados como positivos
		double FN = 0; //quantidade de positivos que foram classificados como negativos
		
		String[] classValues = getClassValues(instancias);
		
		for (int k = 0; k < instancias.numInstances(); k++) {
			
			//System.out.println("Número da instância: " + k);
			Instance instancia = instancias.get(k);
			
			//System.out.println("Classe real: " + k);
			String classeReal = instancias.classAttribute().value((int) instancias.instance(k).classValue());
			
			String classeInferida = null;
			
			//Seta as entradas
			//Considerando que o último atributo � sempre o atributo que corresponde a classe da instância, por isso usa-se o -1
			for (int i = 0; i < (instancias.numAttributes() - 1); i++) {
				
				if(instancias.attribute(i).isNumeric()){
	
					String nomeAtributo = instancias.attribute(i).name();
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
				
				int[] cont = getContadorInt(classValues.length);
				double[] sum = getContadorDouble(classValues.length);
				
				int contRegrasAtivadas = 0;
				
			    for(Rule r : fis.getFunctionBlock(SistemaFuzzy.FUNCTION_BLOCK_NAME).getFuzzyRuleBlock(SistemaFuzzy.RULE_BLOCK_NAME).getRules()){
			    	
			    	double grau = r.getDegreeOfSupport();
			    	String classe = r.getConsequents().getFirst().getTermName();
			    	
			    	if(grau > 0){ //Se a regra foi ativada
			    		
			    		contRegrasAtivadas++;
				    	//System.out.println("indice: "+ k +" | grau: " + grau + " | classe: " + classe);
			    		
//				    	if(classe.equals("positive")){
//				    		
//				    		contPositive++;
//				    		sumPositive += grau;
//				    		
//				    	}
//				    	else if(classe.equals("negative")){
//				    		
//				    		contNegative++;
//				    		sumNegative += grau;
//				    		
//				    	}
			    		

//			    		System.out.println("classe: " + classe);
			    		for (int i = 0; i < classValues.length; i++) {
//			    			System.out.println("	possibilidade: " + classValues[i]);
							if(classe.equals(classValues[i])){
//								System.out.println("		mactch!: " + classValues[i]);
								cont[i]++;
								sum[i] += grau;
								break;
							}
						}

				    	
			    	}
			    	
			    }
			    
//			    double mediaPositive = sumPositive/contPositive;
//			    double mediaNegative = sumNegative/contNegative;
//			    
//			    //System.out.println("sum negative: " + sumNegative + " | contNegative: " + contNegative);
//			    //System.out.println("Quantidade de regras ativadas: " + contRegrasAtivadas);
//			    //System.out.println("Media positive: " + mediaPositive);
//			    //System.out.println("Media negative: " + mediaNegative);
//			    
//			    if(mediaPositive >= mediaNegative){
//			    	
//			    	classeInferida = "positive";
//			    	
//			    }
//			    else{
//			    	
//			    	classeInferida = "negative";
//			    	
//			    }  
			    
			    //calcula as medias
//			    System.out.println(classValues.length);
//			    System.exit(0);
			    double[] medias = getContadorDouble(classValues.length);
			    for (int i = 0; i < classValues.length; i++) {
			    	if(cont[i] != 0){
			    		medias[i] = sum[i]/cont[i];
			    	}
				}
			    
			    //verifica a maior media para inferir a classe
			    double maiorMedia = -1;
			    int classeInferidaIndex = -1;
			    for (int i = 0; i < medias.length; i++) {
//			    	System.out.println("medias "+i+": " + medias[i]);
					if(medias[i] > maiorMedia){
						maiorMedia = medias[i];
						classeInferidaIndex = i;
					}
				}
			    classeInferida = classValues[classeInferidaIndex];
				
			}
			else if(inferenceType == SistemaFuzzy.INFERENCIA_CLASSICA){
				
				/************************** FUZZY CLSSICO *****************************/
				
//				if(fis.getVariable(classAttribute).getValue() >= 0){  // -1 -> negative / 1 -> positive
//					
//					classeInferida = "positive";
//					
//				}
//				else {
//					
//					classeInferida = "negative";
//					
//				}
				
				/**********************************************************************/
				
			}
			
			
//			if(classeInferida.equals("positive")){
//				
//				if(classeReal.equals("positive")){
//					TP++;
//				}
//				else if(classeReal.equals("negative")){
//					FP++;
//				}
//				
//			}
//			else if(classeInferida.equals("negative")){
//				
//				if(classeReal.equals("negative")){
//					TN++;
//				}
//				else if(classeReal.equals("positive")){
//					FN++;
//				}
//				
//			}
			
			
//			System.out.println(classeInferida);
//			System.out.println(classeReal);
//			System.out.println("-------------------------");
//			System.exit(0);
			if(classeInferida.equals(classeReal)){
				TP++;
			}
			//PAREI AQUI: a acurácia ta virada no custipiu dando 100%. algo de errado não está certo

		}
		
		//System.out.println("	Acertos: " + (TP + TN));
		//System.out.println("	Erros: " + (FP + FN));
		
		double acuracia = (TP + TN)/instancias.numInstances();
		//System.out.println("Acurácia: " + acuracia);
		
		return acuracia;
		
	}
	
	private FunctionBlock generateFunctionBlock(Instances instancias, WangMendel wm){
		
		//System.out.println("* Gerando Function Block...");
		
		try {
			
			FunctionBlock functionBlock = new FunctionBlock(fis);
			functionBlock.setName(SistemaFuzzy.FUNCTION_BLOCK_NAME);
			
			HashMap<String, Variable> variaveis = getAtributos(instancias);
			
			//Configurando o bloco de variáveis
			functionBlock.setVariables(variaveis);
			
			//Configurando o bloco de regras
			RuleBlock ruleBlock = wm.generateRuleBlock(SistemaFuzzy.RULE_BLOCK_NAME, functionBlock, variaveis);
			
			//System.out.println("	Tamanho da base de regras: " + ruleBlock.getRules().size());
			
			HashMap<String, RuleBlock> ruleBlocks = new HashMap<String, RuleBlock>();
			ruleBlocks.put(SistemaFuzzy.RULE_BLOCK_NAME, ruleBlock);
			functionBlock.setRuleBlocks(ruleBlocks);
		    
		    //System.out.println("* Function Block gerado com sucesso");
			
			return functionBlock;
			
		} catch (Exception e) {
			
			e.printStackTrace();
			return null;
		}

		
	}
	
	private HashMap<String, Variable> getAtributos(Instances instancias) throws Exception{
		
		HashMap<String, Variable> variables = new HashMap<String, Variable>();
		
		for (int i = 0; i < instancias.numAttributes(); i++) {
			
			Attribute attribute = instancias.attribute(i);
			String nomeAtributo = attribute.name();
			
			if(nomeAtributo.equals(classAttribute)){ //variavel de saida
				
				Variable output_variable = new Variable(nomeAtributo);
				
				for (int j = 0; j < attribute.numValues(); j++) {				
					Value value = new Value();
					//value.setValReal(j);
					output_variable.add(new LinguisticTerm(attribute.value(j), new MembershipFunctionSingleton(value)));	
				}
				
//				Value negative_value = new Value();
//				negative_value.setValReal(-1);
//				MembershipFunctionSingleton especificacoes_termo_negativo = new MembershipFunctionSingleton(negative_value);
//				LinguisticTerm conjuntoFuzzyNegativo = new LinguisticTerm("negative", especificacoes_termo_negativo);
//				output_variable.add(conjuntoFuzzyNegativo);
//				
//				Value positive_value = new Value();
//				positive_value.setValReal(1);
//				MembershipFunctionSingleton especificacoes_termo_positivo = new MembershipFunctionSingleton(positive_value);
//				LinguisticTerm conjuntoFuzzyPositivo = new LinguisticTerm("positive", especificacoes_termo_positivo);
//				output_variable.add(conjuntoFuzzyPositivo);
				
				//Centre of Gravity for Singletons
				DefuzzifierCenterOfGravitySingletons defuzzifier = new DefuzzifierCenterOfGravitySingletons(output_variable);
				
				output_variable.setDefuzzifier(defuzzifier);
				
				variables.put(nomeAtributo, output_variable);
				
				
			}
			else{ //variaveis de entrada
				
				if(attribute.isNumeric()){
					
					AttributeStats as = instancias.attributeStats(i);
					Stats s = as.numericStats;	
					
					/*System.out.println("Atributo: " + instancias.attribute(i).name());
					System.out.println("Valor minimo: " + s.min);
					System.out.println("Valor maximo: " + s.max);*/		

					Variable variable = buildVariavel(nomeAtributo, s.min, s.max);
					variables.put(nomeAtributo, variable);
					
				}
				
			}
			
		}	
		
		return variables;
		
	}
	
	private Variable buildVariavel(String name, double min, double max){
		
		Variable variavel_linguistica = new Variable(name);
		
		HashMap<String, LinguisticTerm> conjuntosFuzzy = new HashMap<String, LinguisticTerm>();
		
		double tamanhoDominio = Math.abs(max - min);
		
		double range = tamanhoDominio/(quantConjuntosFuzzy - 1);
		
		double inf = min - range;
		double sup = min + range;
	
		//Definicao dos limites das regioes de pertinencia triangular
		for(int i = 0; i < quantConjuntosFuzzy; i++){
			
			Value ponto_minimo = new Value();
			ponto_minimo.setValReal(inf);
			
			Value ponto_medio = new Value();
			ponto_medio.setValReal((sup + inf)/2);
			
			Value ponto_maximo = new Value();
			ponto_maximo.setValReal(sup);
			
			MembershipFunctionTriangular especificacoes_termo = new MembershipFunctionTriangular(ponto_minimo, ponto_medio, ponto_maximo);
				
			String nomeConjunto = new String(name + "_" + i);
			
			LinguisticTerm conjuntoFuzzy = new LinguisticTerm(nomeConjunto, especificacoes_termo);
			
			conjuntosFuzzy.put(nomeConjunto, conjuntoFuzzy);
			
			variavel_linguistica.add(conjuntoFuzzy);

			inf += range;
			sup += range;	

		}
		
		variavel_linguistica.setLinguisticTerms(conjuntosFuzzy);
		
		return variavel_linguistica;
		
	}
	
	private String[] getClassValues(Instances test){
		
		Attribute attribute = test.attribute(test.classIndex());
		
		String[] values = new String[attribute.numValues()];
		
		for (int j = 0; j < attribute.numValues(); j++) {				
			values[j] = attribute.value(j);
		}
		
		return values;
	}
	
	
	private int[] getContadorInt(int size){
		int[] array = new int[size];
		for (int i = 0; i < array.length; i++) {
			array[i] = 0;
		}
		return array;
	}
	
	private double[] getContadorDouble(int size){
		double[] array = new double[size];
		for (int i = 0; i < array.length; i++) {
			array[i] = 0;
		}
		return array;
	}

}
