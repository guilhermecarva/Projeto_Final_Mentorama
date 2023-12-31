Neste projeto vamos fazer um web scraping do site yahoo finance de um ativo a sua escolha. O período / time frame tambem pode ser escolhido. Por default o time frame é 1 dia.
Por fim você poderá escolher quantos períodos a frente deseja prever o preço do ativo. Sendo no máximo 10 períodos a frente.
Após pressionar o botão "PREVER", será informado o score R2 obtido no modelo de regressão linear, o preço predito e a última cotação do ativo escolhido.

Um gráfico com o histórico de cotações será plotado e algumas informações financeira, caso esteja disponível (ações/empresas).

Com este projeto viso demonstrar que nem sempre uma modelagem mais complexa é necessária para atingirmos o resultado esperado. Neste caso uma simples regressão linear é o suficiente.
O que faz com que possamos treinar o modelo just in time. Ou seja, para cada ativo e período escolhido. Visto que cada ativo se comprta de uma maneira peculiar.
Basta que tenhamos entendimento das feautures disponíveis. Para este projeto utilizei as seguintes feutures :

-open
-high
-low
-Média móvel de 21 períodos
-Hull Moving Average de 9 períodos
-RSI - Relative Strengh Index de 14 períodos

E o pulo do gato é entender que todos este indicadores utilizam o preço de fechamento dos períodos anteriores em seus cálculos.
Portanto treinamos o modelo com o preço de fechamento deslocado x períodos a frente. 
