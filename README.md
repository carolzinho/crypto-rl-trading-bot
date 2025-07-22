# Bot de Trading Crypto RL com PPO & Streamlit usando Alpaca API 

Este projeto é um bot inteligente de trading de criptomoedas, desenvolvido com aprendizado de máquina por reforço profundo PPO e visualização interativa em Streamlit. Ele visa explorar como modelos de IA podem aprender estratégias de compra e venda utilizando indicadores técnicos reais e métricas quantitativas.

# Features

- Coleta de dados reais de mercado via Alpaca API
- Cálculo automático de indicadores técnicos com ta **indicadores técnicos com valores de parâmetros padrão 
- Treinamento de agente PPO com Stable Baselines3 e ambiente Gym
- Métricas de performance: retorno, Sharpe ratio, drawdown
- Visualização com Streamlit e Matplotlib

# Tecnologias utilizadas
- Dados de mercado: alpaca-py
- Análise técnica: ta
- Bibliotecas de RL: gymnasium e stable-baselines3
- Visualização: matplotlib, streamlit
- Pré-processamento de dados: pandas, numpy, sklearn

# Estrutura do Projeto
- bott.py # main coe com coleta, treinamento e visualização
- modelo_ppo_resultados.zip # modelo PPO treinado com resultados 
- requirements.txt # dependências necessárias
- README.md # esse arquivo

# Como executar localmente

```bash
git clone git clone https://github.com/carolzinho/crypto-rl-trading-bot.git

pip install -r requirements.txt

streamlit run bott.py
