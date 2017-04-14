package sistemaFuzzy;

import java.util.HashMap;
import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.defuzzifier.DefuzzifierCenterOfGravitySingletons;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunctionSingleton;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunctionTriangular;
import net.sourceforge.jFuzzyLogic.membership.Value;
import net.sourceforge.jFuzzyLogic.rule.LinguisticTerm;
import net.sourceforge.jFuzzyLogic.rule.RuleBlock;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.experiment.Stats;

public class SistemaFuzzy {
	
	private Instances instancias;
	private int quantConjuntosFuzzy;
	private String classAttribute;
	private FIS fis;
	
	public static final java.lang.String INFERENCIA_GERAL = "geral";
	public static final java.lang.String INFERENCIA_CLASSICA = "classica";
	
	public SistemaFuzzy(Instances instancias, int quantConjuntosFuzzy, String classAttribute) {
		
		this.instancias = instancias;
		this.quantConjuntosFuzzy = quantConjuntosFuzzy;
		this.classAttribute = classAttribute;
		
		fis = new FIS();
		
	}
	
	public FIS generateFis(String functionBlockName, String ruleBlockName){
		
		
		FunctionBlock fb = generateFunctionBlock(functionBlockName, ruleBlockName);
		fis.addFunctionBlock(functionBlockName, fb);
		
		return fis;
		
	}
	
	private FunctionBlock generateFunctionBlock(String fbName, String rbName){
		
		System.out.println("* Gerando Function Block...");
		
		try {
			
			FunctionBlock functionBlock = new FunctionBlock(fis);
			functionBlock.setName(fbName);
			
			HashMap<String, Variable> variaveis = getAtributos();
			
			//Configurando o bloco de variáveis
			functionBlock.setVariables(variaveis);
			
			//Configurando o bloco de regras
			WangMendel wm = new WangMendel(instancias, classAttribute);
			RuleBlock ruleBlock = wm.generateRuleBlock(rbName, functionBlock, variaveis);
			
			System.out.println("Quantidade de regras geradas: " + ruleBlock.getRules().size());
			
			HashMap<String, RuleBlock> ruleBlocks = new HashMap<String, RuleBlock>();
			ruleBlocks.put(rbName, ruleBlock);
			functionBlock.setRuleBlocks(ruleBlocks);
		    
		    System.out.println("* Function Block gerado com sucesso");
			
			return functionBlock;
			
		} catch (Exception e) {
			
			e.printStackTrace();
			return null;
		}

		
	}
	
	private HashMap<String, Variable> getAtributos() throws Exception{
		
		HashMap<String, Variable> variables = new HashMap<String, Variable>();
		
		//Considerando que o último atributo é sempre o atributo que corresponde a classe da instância, por isso usa-se o -1
		for (int i = 0; i < this.instancias.numAttributes(); i++) {
			
			String nomeAtributo = instancias.attribute(i).name();
			
			if(nomeAtributo.equals(classAttribute)){ //variável de saída
				
				Variable output_variable = new Variable(nomeAtributo);
				
				Value negative_value = new Value();
				negative_value.setValReal(-1);
				MembershipFunctionSingleton especificacoes_termo_negativo = new MembershipFunctionSingleton(negative_value);
				LinguisticTerm conjuntoFuzzyNegativo = new LinguisticTerm("negative", especificacoes_termo_negativo);
				output_variable.add(conjuntoFuzzyNegativo);
				
				Value positive_value = new Value();
				positive_value.setValReal(1);
				MembershipFunctionSingleton especificacoes_termo_positivo = new MembershipFunctionSingleton(positive_value);
				LinguisticTerm conjuntoFuzzyPositivo = new LinguisticTerm("positive", especificacoes_termo_positivo);
				output_variable.add(conjuntoFuzzyPositivo);
				
				//Centre of Gravity for Singletons
				DefuzzifierCenterOfGravitySingletons defuzzifier = new DefuzzifierCenterOfGravitySingletons(output_variable);
				
				output_variable.setDefuzzifier(defuzzifier);
				
				variables.put(nomeAtributo, output_variable);
				
				
			}
			else{ //variáveis de entrada
				
				if(instancias.attribute(i).isNumeric()){
					
					AttributeStats as = instancias.attributeStats(i);
					Stats s = as.numericStats;	
					
					/*System.out.println("Atributo: " + instancias.attribute(i).name());
					System.out.println("Valor mínimo: " + s.min);
					System.out.println("Valor máximo: " + s.max);*/		

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
	
		//Definição dos limites das regiões de pertinencia triangular
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

}
