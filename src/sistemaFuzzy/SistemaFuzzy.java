package sistemaFuzzy;

import java.util.HashMap;
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
import utils.Utils;
import utils.WekaUtils;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;

public class SistemaFuzzy {
	
	private static int quantConjuntosFuzzy = 3;
	private FIS jFuzzyLogic;
	private String inferenceType;
	private long tempoInicial; //tempo inicial da execução do algoritmo
	private long tempoFinal; //tempo final da executação do algoritmo
	private int tamanhoBaseRegras = 0;
	
	public static final java.lang.String INFERENCIA_GERAL = "geral";
	public static final java.lang.String INFERENCIA_CLASSICA = "classica";
	
	public static final java.lang.String FUNCTION_BLOCK_NAME = "functionBlock";
	public static final java.lang.String RULE_BLOCK_NAME = "ruleBlock";
	
	public SistemaFuzzy(String inferenceType){
		this.inferenceType = inferenceType;
		jFuzzyLogic = new FIS();
	}

	public long getTempoInicial() {
		return tempoInicial;
	}

	public void setTempoInicial(long tempoInicial) {
		this.tempoInicial = tempoInicial;
	}

	public long getTempoFinal() {
		return tempoFinal;
	}

	public void setTempoFinal(long tempoFinal) {
		this.tempoFinal = tempoFinal;
	}

	/**
	 * Executa e retorna a acurácia do sistema fuzzy
	 * @param train Instâncias de treinamento para geração da base de regras
	 * @param test Instâncias de teste
	 * @return String[] Retorna um array de strings contendo as a classe inferida para instância de teste 
	 * @throws Exception
	 */
	public String[] run(Instances train, Instances test) throws Exception{
		FunctionBlock fb = generateFunctionBlock(train, new WangMendel(train));
		return test(fb, test);
	}
	
	/**
	 * Retorna um array de strings contendo a classe inferida para cada instância de teste. Neste caso, a classe da base de dados testada precisa necessariamente ser nominal
	 * @param functionBlock
	 * @param test
	 * @return
	 */
	private String[] test(FunctionBlock functionBlock, Instances test){
				
		jFuzzyLogic.addFunctionBlock(SistemaFuzzy.FUNCTION_BLOCK_NAME, functionBlock);
				
		String[] classValues = getClassValues(test);
		String[] classesInferidas = new String[test.size()];
		
		for (int k = 0; k < test.numInstances(); k++) {
			
			//System.out.println("Número da instância: " + k);
			Instance instancia = test.get(k);
			
			//System.out.println("Classe real: " + k);
			String classeReal = test.classAttribute().value((int) test.instance(k).classValue());
			
			String classeInferida = null;
			
			//Seta as entradas
			//Considerando que o último atributo � sempre o atributo que corresponde a classe da instância, por isso usa-se o -1
			for (int i = 0; i < (test.numAttributes() - 1); i++) {
				
				if(test.attribute(i).isNumeric()){
	
					String nomeAtributo = test.attribute(i).name();
					double valor = instancia.value(i);
					//System.out.println(nomeAtributo + ": " + valor);
					
					jFuzzyLogic.setVariable(nomeAtributo, valor);
				}	
			}
	
			// Evaluate
			jFuzzyLogic.evaluate();
			
			if(inferenceType == SistemaFuzzy.INFERENCIA_GERAL){
				
				/************************** FUZZY GERAL *****************************/
				
				int[] cont = getContadorInt(classValues.length);
				double[] sum = getContadorDouble(classValues.length);
				
			    for(Rule r : jFuzzyLogic.getFunctionBlock(SistemaFuzzy.FUNCTION_BLOCK_NAME).getFuzzyRuleBlock(SistemaFuzzy.RULE_BLOCK_NAME).getRules()){
			    	
			    	double grau = r.getDegreeOfSupport();
			    	String classe = r.getConsequents().getFirst().getTermName();
			    	
			    	if(grau > 0){ //Se a regra foi ativada
				    	//System.out.println("indice: "+ k +" | grau: " + grau + " | classe: " + classe)
			    		
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
				
				/************************** FUZZY CLASSICO *****************************/
				
//				if(jFuzzyLogic.getVariable(classAttribute).getValue() >= 0){  // -1 -> negative / 1 -> positive
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
			
//			System.out.println(classeInferida);
//			System.out.println(classeReal);
//			System.out.println("-------------------------");
//			System.exit(0);
			
			classesInferidas[k] = classeInferida;

		}
		
		return classesInferidas;
		
	}
	
	private FunctionBlock generateFunctionBlock(Instances instancias, WangMendel wm){
		
		//System.out.println("* Gerando Function Block...");
		
		try {
			
			FunctionBlock functionBlock = new FunctionBlock(jFuzzyLogic);
			functionBlock.setName(SistemaFuzzy.FUNCTION_BLOCK_NAME);
			
			HashMap<String, Variable> variaveis = getAtributos(instancias);
			
			//Configurando o bloco de variáveis
			functionBlock.setVariables(variaveis);
			
			//Configurando o bloco de regras
			RuleBlock ruleBlock = wm.generateRuleBlock(SistemaFuzzy.RULE_BLOCK_NAME, functionBlock, variaveis);
			this.tamanhoBaseRegras = ruleBlock.getRules().size();
			
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
		String classAttribute = WekaUtils.getClassAttributeName(instancias);
		
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
	
	public int getamanhoBaseRegras(){
		return this.tamanhoBaseRegras;
	}

}
