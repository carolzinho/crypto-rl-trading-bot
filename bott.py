import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import ta
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# configurações iniciais
CHAVE_API_ALPACA = "PK7R6H4AP485UYTHD387"
CHAVE_SECRETA_ALPACA = "0xU7xZyusp4rbaM7D4GWHgsAhvgvgRh7sDFd7ubU"
capital_inicial = 10000
risco_por_trade = 0.01

# cliente de dados históricos
dados_client = CryptoHistoricalDataClient(CHAVE_API_ALPACA, CHAVE_SECRETA_ALPACA)

simbolo = "BTC/USD"
periodo_tempo = TimeFrame.Minute

# função de coleta e cálculo de indicadores com barras
def buscar_dados():
    fim = datetime.utcnow()
    inicio = fim - timedelta(days=7)

    requisicao_barras = CryptoBarsRequest(
        symbol_or_symbols=simbolo,
        timeframe=periodo_tempo,
        start=inicio,
        end=fim
    )
    barras = dados_client.get_crypto_bars(requisicao_barras).df

    if barras.empty:
        raise ValueError("nenhum dado retornado pela API.")

    df = barras.copy()
    df.index = pd.to_datetime(df.index.get_level_values('timestamp'))
    df = df[~df.index.duplicated(keep='first')]

    if len(df) < 100:
        raise ValueError("dados insuficientes para calcular indicadores.")

    df['retornos'] = df['close'].pct_change().fillna(0)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(0)
    df['macd'] = ta.trend.MACD(df['close']).macd().fillna(0)
    df['sma'] = ta.trend.sma_indicator(df['close'], window=20).fillna(0)
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_alta'] = bb.bollinger_hband().fillna(0)
    df['bb_baixa'] = bb.bollinger_lband().fillna(0)
    df['ema_rapida'] = ta.trend.ema_indicator(df['close'], window=8).fillna(0)
    df['ema_lenta'] = ta.trend.ema_indicator(df['close'], window=21).fillna(0)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close']).fillna(0)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']).fillna(0)
    df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc().fillna(0)
    df['willr'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r().fillna(0)
    df['forca_tendencia'] = (df['ema_rapida'] > df['ema_lenta']).astype(int)
    df['retorno_1'] = df['close'].pct_change(1).fillna(0)
    df['momentum_3'] = df['close'].pct_change(3).fillna(0)
    df['volatilidade_3'] = df['retornos'].rolling(3).std().fillna(0)
    df['media_volume_5'] = df['volume'].rolling(5).mean().fillna(0)
    df['spread'] = 0

    colunas_normalizar = ['rsi', 'macd', 'sma', 'bb_alta', 'bb_baixa',
                          'ema_rapida', 'ema_lenta', 'adx', 'atr', 'roc', 'willr',
                          'retorno_1', 'momentum_3', 'volatilidade_3', 'media_volume_5']

    df = df.dropna(subset=colunas_normalizar)

    if df.empty:
        raise ValueError("nenhum dado com todos os indicadores disponíveis para normalização.")

    scaler = StandardScaler()
    df[colunas_normalizar] = scaler.fit_transform(df[colunas_normalizar])

    return df

@st.cache_data
def carregar_dados():
    return buscar_dados()

df = carregar_dados()
class AmbienteTrading(gym.Env):
    def __init__(self, df, janela=10):
        super(AmbienteTrading, self).__init__()
        self.df = df.reset_index()
        self.janela = janela
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(janela, 23), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.passo_atual = np.random.randint(self.janela, len(self.df) - 100)
        self.saldo = capital_inicial
        self.crypto_possuida = 0
        self.valor_total = self.saldo
        self.valor_total_anterior = self.valor_total
        self.preco_entrada = 0
        return self._obter_observacao(), {}

    def _obter_observacao(self):
        janela_df = self.df.iloc[self.passo_atual - self.janela:self.passo_atual]
        obs = []
        for _, linha in janela_df.iterrows():
            obs.append([
                linha['open'], linha['high'], linha['low'], linha['close'], linha['volume'],
                linha['rsi'], linha['macd'], linha['sma'], linha['bb_alta'], linha['bb_baixa'],
                linha['ema_rapida'], linha['ema_lenta'], linha['adx'], linha['atr'],
                linha['roc'], linha['willr'], linha['forca_tendencia'],
                linha['retorno_1'], linha['momentum_3'], linha['volatilidade_3'],
                linha['media_volume_5'], linha['spread'],
                int(self.crypto_possuida > 0)
            ])
        return np.array(obs, dtype=np.float32)

    def step(self, acao):
        concluido = False
        info = {}
        linha = self.df.iloc[self.passo_atual]
        preco = linha['close']
        atr = linha['atr']
        rsi = linha['rsi']
        forca_tendencia = linha['forca_tendencia']

        tp = 3.0 * atr if forca_tendencia else 0.5 * atr
        sl = 2.0 * atr if rsi < 0 else 1.0 * atr

        risco_maximo = self.saldo * risco_por_trade
        quantidade = risco_maximo / preco if preco > 0 else 0

        recompensa = 0

        if acao == 1 and self.saldo >= preco * quantidade:
            self.crypto_possuida += quantidade
            self.saldo -= preco * quantidade
            self.preco_entrada = preco
            recompensa -= 0.01  # penaliza manter posição aberta sem vender

        elif acao == 2 and self.crypto_possuida > 0:
            lucro = (preco - self.preco_entrada) * self.crypto_possuida
            self.saldo += preco * self.crypto_possuida
            self.crypto_possuida = 0
            self.preco_entrada = 0
            recompensa += lucro

        elif acao == 0:
            recompensa -= 0.005

        lucro_nao_realizado = (preco - self.preco_entrada) * self.crypto_possuida if self.crypto_possuida > 0 else 0

        if self.crypto_possuida > 0:
            if lucro_nao_realizado >= tp * self.crypto_possuida or lucro_nao_realizado <= -sl * self.crypto_possuida:
                self.saldo += preco * self.crypto_possuida
                self.crypto_possuida = 0
                self.preco_entrada = 0
                recompensa += lucro_nao_realizado

        self.valor_total = self.saldo + (self.crypto_possuida * preco)
        recompensa += self.valor_total - self.valor_total_anterior
        self.valor_total_anterior = self.valor_total

        self.passo_atual += 1
        if self.passo_atual >= len(self.df) - 1:
            concluido = True

        return self._obter_observacao(), recompensa, concluido, False, info

# treino do agente com PPO
ambiente = AmbienteTrading(df)
modelo = PPO("MlpPolicy", ambiente, verbose=0, ent_coef=0.01)
modelo.learn(total_timesteps=500000)
modelo.save("modelo_ppo_daytrade")

# avaliação do modelo treinado
observacao, _ = ambiente.reset()
recompensas, valores, acoes, retornos = [], [], [], []
pico = -np.inf
max_drawdown = 0
precos_plot = []
datas_plot = []

for _ in range(len(df) - 100):
    acao, _ = modelo.predict(observacao)
    acoes.append(acao)
    linha = ambiente.df.iloc[ambiente.passo_atual]
    precos_plot.append(linha['close'])
    datas_plot.append(linha['timestamp'])
    observacao, recompensa, concluido, _, _ = ambiente.step(acao)
    recompensas.append(recompensa)
    valores.append(ambiente.valor_total)
    retornos.append(recompensa)
    pico = max(pico, ambiente.valor_total)
    drawdown = pico - ambiente.valor_total
    max_drawdown = max(max_drawdown, drawdown)
    if concluido:
        break

retornos = np.array(retornos)
indice_sharpe = (np.mean(retornos) / (np.std(retornos) + 1e-9)) * np.sqrt(252) if np.std(retornos) != 0 else 0

# visualizações
st.title("performance final:")

st.subheader("evolução do capital")
fig1, ax1 = plt.subplots()
ax1.plot(datas_plot, valores, label="capital")
ax1.set_xlabel("tempo")
ax1.set_ylabel("capital em usdt")
ax1.tick_params(axis='x', rotation=45)
ax1.legend()
st.pyplot(fig1)

st.subheader("ações sobre o preço")
fig2, ax2 = plt.subplots()
ax2.plot(datas_plot, precos_plot, label="preço", color='gray')
ax2.plot([datas_plot[i] for i, a in enumerate(acoes) if a == 1],
         [precos_plot[i] for i, a in enumerate(acoes) if a == 1],
         '^', color='green', label='compra')
ax2.plot([datas_plot[i] for i, a in enumerate(acoes) if a == 2],
         [precos_plot[i] for i, a in enumerate(acoes) if a == 2],
         'v', color='red', label='venda')
ax2.set_title("ações tomadas sobre o gráfico de candles")
ax2.tick_params(axis='x', rotation=45)
ax2.legend()
st.pyplot(fig2)

st.subheader("retornos acumulados")
fig3, ax3 = plt.subplots()
ax3.plot(datas_plot, np.cumsum(retornos), color='green')
ax3.set_xlabel("tempo")
ax3.set_ylabel("retorno acumulado")
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

st.subheader("métricas de performance")
st.write(f"capital final: {round(valores[-1], 2)} usdt")
st.write(f"lucro total: {round(valores[-1] - capital_inicial, 2)} usdt")
st.write(f"retorno médio: {round(np.mean(retornos), 5)}")
st.write(f"volatilidade: {round(np.std(retornos), 5)}")
st.write(f"índice de sharpe: {round(indice_sharpe, 3)}")
st.write(f"máximo drawdown: {round(max_drawdown, 2)} usdt")
