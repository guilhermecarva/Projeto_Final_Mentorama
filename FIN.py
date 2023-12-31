import pandas as pd
from yahooquery import Ticker
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime
import time
import warnings
import streamlit as st
import locale
from PIL import Image
import plotly.express as px
warnings.filterwarnings("ignore")

locale.setlocale(locale.LC_ALL, '')
st.set_page_config(page_title='Mentorama Projeto Final', layout='wide', page_icon=':chart_with_upwards_trend:')

image = Image.open("mentorama.png")
image = image.convert("RGB")
st.image(image, output_format='auto')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}<style>', unsafe_allow_html=True)

def hma(period,imp_data):
 wma_1 = imp_data['close'].rolling(period//2).apply(lambda x: \
 np.sum(x * np.arange(1, period//2+1)) / np.sum(np.arange(1, period//2+1)), raw=True)
 wma_2 = imp_data['close'].rolling(period).apply(lambda x: \
 np.sum(x * np.arange(1, period+1)) / np.sum(np.arange(1, period+1)), raw=True)
 diff = 2 * wma_1 - wma_2
 hma = diff.rolling(int(np.sqrt(period))).mean()
 return hma

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def importando(ticker, intervalo):
  dados = Ticker(ticker)
  if intervalo == '1d':
    dados = dados.history(period='6000d', interval = intervalo)
  elif intervalo == '1wk':
    dados = dados.history(period='7000d', interval = intervalo)
  elif intervalo == '1mo':
    dados = dados.history(period='8000d', interval = intervalo)
  else:
    dados = dados.history(interval = intervalo)
  dados = dados.droplevel(level=0)
  dados = dados.reset_index()
  dados['date'] = pd.to_datetime(dados['date'], format='%Y-%m-%d')
  return dados

def prevendo(dados,futuro):
  real = dados.copy().tail(1)
  dados['MM21'] = dados['close'].rolling(21).mean()
  dados['HMA9'] = hma(9,dados)
  dados['RSI'] = calculate_rsi(dados['close'])
  dados['close'] = dados['close'].shift(-int(futuro))
  dados.dropna(inplace=True)
  dados = dados.reset_index(drop=True)
  features = dados[['high', 'low', 'volume', 'MM21', 'HMA9', 'RSI']]
  labels = dados['close']
  linhas = len(features)
  treino= round(.70 * linhas)
  teste= linhas - treino
  valida = linhas -1
  X_train = features[:treino]
  X_test = features[treino:linhas -1]
  y_train = labels[:treino]
  y_test = labels[treino:linhas -1]
  scaler = MinMaxScaler()
  X_train_norm = scaler.fit_transform(X_train)
  X_test_norm  = scaler.transform(X_test)
  reg = LinearRegression().fit(X_train_norm, y_train)
  pred = reg.predict(X_test_norm)
  r2 = r2_score(y_test, pred)
  predito=reg.predict(scaler.transform(features.tail(1)))
  return r2, predito, real

def prever_ticker():

  #st.write('Calculando.....', end = '')
  #time.sleep(1)
  #st.write('\rPrevisões com score inferior a 70% serão descartados ', end = '')
  #time.sleep(2)
  #st.write('\r ...', end = '')

  try:
    imp_data = importando(ticker,intervalo)
    preds = prevendo(imp_data,futuro)

    if preds[0]<0.7:
      st.write('')
      st.write('Desculpe, não foi possível obter uma boa predição deste ticker e/ou período. Por favor escolha outro')
    else:
      col1, col2, col3 = st.columns(3)

      col1.metric(label="Score (%)", value=round(preds[0] * 100,2))
      col2.metric(label=f"Previsão ($) {ticker}", value=round(preds[1].item(),2) )
      col3.metric(label=f"Última cotação real em {str(np.datetime64(preds[2]['date'].values[0], 'D'))}", value=str(round(preds[2]['close'].values[0],2)) )

      st.write('')
      #st.write(f'Com um coeficiente de determinação r2 de {preds[0] * 100:.2f}%')
      #st.write(f'O valor de fechamento do {ticker} previsto para o próximo período é : {preds[1].item():.2f}')
      #st.write(f"A última cotação real foi em {str(np.datetime64(preds[2]['date'].values[0], 'D'))} no valor de de {str(round(preds[2]['close'].values[0],2))}")

  except:
    st.write('')
    st.write('Certifique-se de entrar com um ticker existente e/ou intervalo correto')
    #st.write('Caso tenha dúvidas, consulte https://br.financas.yahoo.com ')

#=============================================================================================================#
#=============================================================================================================#
#=============================================================================================================#
c_1 = st.container()

with c_1:
    st.markdown("<h1 style='text-align: center;'>PROJETO FINAL CIENTISTA DE DADOS</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.markdown("<h2 style='text-align: center;'>Este projeto visa realizar um web scraping com as cotações de um ativo a sua escolha e utilizando \
    machine learning, realizar a previsão de uma cotação futura</h2>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.01)
my_bar.progress(percent_complete + 1)

ticker = st.text_input("Entre com o Ticker que deseja predizer ", "",key="tick")
st.write("Exemplo : PETR4.SA       VALE3.SA       GOOG       APLE      GOLD      BBDC3.SA      ABEV3.SA  BTC-USD   ")
st.write("Você poderá consultar os ativos válidos em https://br.financas.yahoo.com")

st.write("")
st.write("")

intervalo = st.selectbox("Qual o intervalo/time frame das cotações ? ", ["1d", "15m", "1h", "1d", "1wk", "1mo"])
st.write("")
st.write("")
futuro = st.slider("Quantos períodos a frente deseja prever ? ",1, 10,1)

prever = st.button(":chart_with_upwards_trend: PREVER :chart_with_downwards_trend:")
if prever:
    prever_ticker()
#prever_ticker()

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.01)
my_bar.progress(percent_complete + 1)

#====================================================================#
#====================================================================#

if prever:
    try:
     df = importando(ticker,intervalo)
     df = df.tail(int(len(df)*0.2))
     fig = px.line(df[['date','close']], x='date' ,y='close', title='Cotação histórica')
     st.plotly_chart(fig,theme="streamlit", use_container_width=True)
    except:
     st.write("Não foi possível criar o gráfico")


my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.01)
my_bar.progress(percent_complete + 1)

#====================================================================#
#====================================================================#

def fin_info():
  try:
    fin = Ticker(ticker)
    fin = fin.income_statement()
    index = fin.index
    fin = fin.transpose()
    fin.columns = fin.iloc[0,:]
    fin = fin.iloc[2:,:-1]
    fin = fin.iloc[:, ::-1]
    st.write(f'Informações financeiras do ticker : {index[0]}')
    return fin
  except:
    st.write('Não foi possível obter as informações financeiras')

if prever:
    st.table(fin_info())



