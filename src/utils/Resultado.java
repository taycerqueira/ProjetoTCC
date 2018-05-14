package utils;

public class Resultado {
	
	private int qtdAcertos;
	private long tempoInicial;
	private long tempoFinal;
	private int qtdInstanciasAntes;
	private int qtdInstanciasTreinamento;
	private int qtdInstanciasTeste;
	private int qtdRegras;
	private int fold;
	
	public Resultado(int qtdInstanciasTreinamento, int qtdInstanciasTeste) {
		this.qtdInstanciasTreinamento = qtdInstanciasTreinamento;
		this.qtdInstanciasTeste = qtdInstanciasTeste;
		qtdAcertos = 0;
		qtdInstanciasAntes = 0;
		qtdRegras = 0;
	}

	public int getQtdAcertos() {
		return qtdAcertos;
	}

	public void setQtdAcertos(int qtdAcertos) {
		this.qtdAcertos = qtdAcertos;
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
	
	public int getQtdInstanciasAntes() {
		return qtdInstanciasAntes;
	}

	public void setQtdInstanciasAntes(int qtdInstanciasAntes) {
		this.qtdInstanciasAntes = qtdInstanciasAntes;
	}
	
	public int getQtdInstanciasTeste() {
		return qtdInstanciasTeste;
	}

	public void setQtdInstanciasTeste(int qtdInstanciasTeste) {
		this.qtdInstanciasTeste = qtdInstanciasTeste;
	}

	public int getFold(){
		return this.fold;
	}

	public void setFold(int fold) {
		this.fold = fold;
	}

	public int getQtdInstancias() {
		return qtdInstanciasTreinamento;
	}
	
	public void setQtdInstancias(int qtdInstancias) {
		this.qtdInstanciasTreinamento = qtdInstancias;
	}
	
	public int getQtdRegras() {
		return qtdRegras;
	}
	
	public void setQtdRegras(int qtdRegras) {
		this.qtdRegras = qtdRegras;
	}
	
	public void startTime(){
		this.tempoInicial = System.currentTimeMillis();
	}
	
	public void endTime(){
		this.tempoFinal  = System.currentTimeMillis();
	}
	
	public long getDuracao(){
		return tempoFinal - tempoInicial;
	}
	
	public double calcularAcuracia(){
		return (double)qtdAcertos/(double)qtdInstanciasTeste;
	}
	
	public double calcularReducao(){
		if(qtdInstanciasAntes == 0){
			return 0;
		}
		else{
			return (qtdInstanciasAntes - qtdInstanciasTreinamento) / (double) qtdInstanciasAntes;
		}
	}

}
