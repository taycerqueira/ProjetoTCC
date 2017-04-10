package wangMendel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.FunctionBlock;
import net.sourceforge.jFuzzyLogic.defuzzifier.DefuzzifierCenterOfGravitySingletons;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunction;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunctionSingleton;
import net.sourceforge.jFuzzyLogic.membership.MembershipFunctionTriangular;
import net.sourceforge.jFuzzyLogic.membership.Value;
import net.sourceforge.jFuzzyLogic.rule.LinguisticTerm;
import net.sourceforge.jFuzzyLogic.rule.Rule;
import net.sourceforge.jFuzzyLogic.rule.RuleBlock;
import net.sourceforge.jFuzzyLogic.rule.RuleExpression;
import net.sourceforge.jFuzzyLogic.rule.RuleTerm;
import net.sourceforge.jFuzzyLogic.rule.Variable;
import net.sourceforge.jFuzzyLogic.ruleAccumulationMethod.RuleAccumulationMethodMax;
import net.sourceforge.jFuzzyLogic.ruleActivationMethod.RuleActivationMethodMin;
import net.sourceforge.jFuzzyLogic.ruleConnectionMethod.RuleConnectionMethodAndMin;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;


public class WangMendel {
	
	private DataSource dados;
	private Instances instancias;
	private int[] indicesTeste;
	
	
	public WangMendel(DataSource dados, int[] indicesTeste) throws Exception{
		
		this.dados = dados;
		this.instancias = dados.getDataSet();
		this.indicesTeste = indicesTeste;

	}
	
	public FunctionBlock generateFunctionBlock(FIS fis, String name, DataSource dados, int quantRegioes, String output_name){
				
		try {
			
			FunctionBlock functionBlock = new FunctionBlock(fis);
			functionBlock.setName(name);
			
			HashMap<String, Variable> variaveis = getAtributos(dados, quantRegioes, output_name);
			
			//Configurando o bloco de variáveis
			functionBlock.setVariables(variaveis);
			
			//Configurando o bloco de regras
			RuleBlock ruleBlock = generateRuleBlock(functionBlock, variaveis, output_name);
			
			HashMap<String, RuleBlock> ruleBlocks = new HashMap<String, RuleBlock>();
			ruleBlocks.put("ruleblock1", ruleBlock);
			functionBlock.setRuleBlocks(ruleBlocks);
			
			return functionBlock;
			
		} catch (Exception e) {
			
			e.printStackTrace();
			return null;
		}

		
	}
	
	public RuleBlock generateRuleBlock(FunctionBlock functionBlock, HashMap<String, Variable> variaveis, String output_name) throws Exception{
		
		RuleBlock ruleBlock = new RuleBlock(functionBlock);
		ruleBlock.setRuleActivationMethod(new RuleActivationMethodMin());
		ruleBlock.setRuleAccumulationMethod(new RuleAccumulationMethodMax());
		
		//Armazena o indice do atributo que corresponde a classe. Aqui considero que é sempre o último atributo.
		int indiceClasse = dados.getDataSet().numAttributes() - 1; 
		
		int contInstancias = 0; 
		for (int k = 0; k < instancias.size(); k++ ) {
			
			//Se a instância não for de teste...
			if(!isParaTeste(k)){
				
				contInstancias++;
				
				Instance instancia = instancias.get(k);
	
				Rule regra = new Rule("Rule " + contInstancias, ruleBlock);
				
				String classeInstancia = instancia.stringValue(indiceClasse);
				Variable output_variable = variaveis.get(output_name);
				
				regra.addConsequent(output_variable, classeInstancia, false);
				
				//Começa com 1 por ser um fator neutro na multiplicação
				double grauRegra = 1; 
				
				RuleExpression antecedents = null;
				RuleTerm fuzzyRuleTerm1 = null;
				ArrayList<LinguisticTerm> antecedentesConjuntosFuzzy = new ArrayList<LinguisticTerm>();
				
				//Pega cada atributo da instância
				for (int i = 0; i < indiceClasse; i++) {
					
					//Para cada atributo da instância, verifico o conjunto fuzzy de maior grau
					double maiorGrau = Double.NEGATIVE_INFINITY;
					LinguisticTerm conjuntoMaiorGrau = null;	
								
					String atributeName = instancia.attribute(i).name();					
					
					double valor = instancia.value(i);
					Variable variavel = variaveis.get(atributeName);
					
					List<LinguisticTerm> conjuntosFuzzy = variavel.linguisticTermsSorted();
					for (LinguisticTerm conjunto : conjuntosFuzzy) {

						MembershipFunction pertinencia = conjunto.getMembershipFunction();
						double grau = pertinencia.membership(valor);
						
						if(grau > maiorGrau){
							maiorGrau = grau;
							conjuntoMaiorGrau = conjunto;
						}
						
					}
					
					antecedentesConjuntosFuzzy.add(conjuntoMaiorGrau);
					
					grauRegra *= maiorGrau;
					
					if(i == 0){
						
						fuzzyRuleTerm1 = new RuleTerm(variavel, conjuntoMaiorGrau.getTermName(), false);
						
					}
					else if(i == 1){
							
						RuleTerm fuzzyRuleTerm2 = new RuleTerm(variavel, conjuntoMaiorGrau.getTermName(), false);
						antecedents = new RuleExpression(fuzzyRuleTerm1, fuzzyRuleTerm2, RuleConnectionMethodAndMin.get());
						
					}
					else{
						//System.out.println(antecedents);
						
						RuleTerm nextFuzzyRuleTerm = new RuleTerm(variavel, conjuntoMaiorGrau.getTermName(), false);
						antecedents = new RuleExpression(antecedents, nextFuzzyRuleTerm, RuleConnectionMethodAndMin.get());
						
					}
					
				}
					
				regra.setDegreeActivationWangMendel(grauRegra);
				regra.setAntecedentesConjuntosFuzzy(antecedentesConjuntosFuzzy);
				regra.setAntecedents(antecedents);		
				
				boolean adiciona_regra = true;
				
				List<Rule> regrasExistentes = ruleBlock.getRules();
				
				//Verificar se existe uma regra já adicionada com os mesmos antecedentes
				for (Rule r : regrasExistentes) {
					
					ArrayList<LinguisticTerm> antecedentes = r.getAntecedentesConjuntosFuzzy();
					
					int cont = 0;
					
					for (LinguisticTerm conjuntoRegraJaExistente : antecedentes) {
						
						for (LinguisticTerm conjuntoRegraAtual : antecedentesConjuntosFuzzy) {
							
							if(conjuntoRegraJaExistente.getTermName().equals(conjuntoRegraAtual.getTermName())){
								cont++;
							}
							
						}
						
					}
					
					if(cont == antecedentes.size()){ //regra em duplicidade. verificar qual pussui maior grau
						
						if(r.getDegreeActivationWangMendel() < regra.getDegreeActivationWangMendel()){
							
							//Remover a regra já existente e adicionar a nova regra
							ruleBlock.remove(r);
							break;
							
						}
						else{
							
							adiciona_regra = false;
							
						}
						
					}
			
				}
				
				if(adiciona_regra){
					ruleBlock.add(regra);
				}
						
			}
			
		}
					
		return ruleBlock;
		
	}

	private HashMap<String, Variable> getAtributos(DataSource dados, int quantRegioes, String output_name) throws Exception{
		
		HashMap<String, Variable> variables = new HashMap<String, Variable>();
		
		//Considerando que o último atributo é sempre o atributo que corresponde a classe da instância, por isso usa-se o -1
		for (int i = 0; i < dados.getDataSet().numAttributes(); i++) {
			
			String nomeAtributo = instancias.attribute(i).name();
			
			if(nomeAtributo.equals(output_name)){ //variável de saída
				
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

					Variable variable = buildVariavel(nomeAtributo, s.min, s.max, quantRegioes);
					variables.put(nomeAtributo, variable);
					
				}
				
			}
			
		}	
		
		return variables;
		
	}
	
	private Variable buildVariavel(String name, double min, double max, int quantRegioes){
		
		Variable variavel_linguistica = new Variable(name);
		
		HashMap<String, LinguisticTerm> conjuntosFuzzy = new HashMap<String, LinguisticTerm>();
		
		double tamanhoDominio = Math.abs(max - min);
		
		double range = tamanhoDominio/(quantRegioes - 1);
		
		double inf = min - range;
		double sup = min + range;
	
		//Definição dos limites das regiões de pertinencia triangular
		for(int i = 0; i < quantRegioes; i++){
			
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
	
	private boolean isParaTeste(int k){
		
		boolean resultado = false;
		for(int i = 0; i < indicesTeste.length; i++){
			if(indicesTeste[i] == k){
				resultado =  true;
				break;
			}
		}
		return resultado;
	}

}
