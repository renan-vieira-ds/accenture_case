# Previsão de Gastos com Plano de Saúde

Este projeto visa prever o valor **gasto anualmente por clientes de um plano de saúde**, com modelos que **penalizam o erro de subestimar os gatos**, uma escolha estratégica para **evitar prejuízos por provisionamento insuficiente**.

## Ambiente Virtual

1. Crie o ambiente virtual:

```bash
python -m venv .venv
```

2. Ative o ambiente:

- Windows:
```bash
.venv\Scripts\activate
```
- Linux/macOS:
```bash
source .venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Notebooks do Projeto

### 1. `eda.ipynb`
- Análise exploratória com:
  - Distribuição de variáveis numéricas e categóricas
  - Scatterplots com `valor`
  - Boxplots por categorias
  - Correlações por faixa de valor
  - Análise com A Priori para  extraçao de regras
- Gera gráficos em: `plots/`

---

### 2. `data_prep.ipynb`
- Pipeline de pré-processamento:
  - Imputação
  - Padronizaçao
  - OneHotEncoding (binário e não binário)
  - ColumnTransformer
- Gera artefato:
  - `artifacts/pipeline_completo.pkl`: pipeline salvo com prepare_data

---

### 3. `feature_selection.ipynb`
- Técnicas aplicadas:
  - **Boruta** com Random Forest Regressor
  - **Importância de features** (Random Forest)
  - **Multicolinearidade** com VIF
- Resultado:
  - Apenas **3 variáveis** selecionadas pelo Boruta
    - Idade, IMC e fumante
    - Boruta é confiável pois cria "shadows" (variáveis barulho) e compara a importância das reais com essas.
    - Se a feature for mais importante que o melhor shadow, é confirmada.
  - Seleçao confirmada  por feature  importance com Random Forest
  - Multicolinearidade confirmou nao haver correlaçao entre as variáveis selecionadas
- Gera gráficos de importância e multicolinearidade em `plots/`
- Como artefato um csv com a seleçao do Boruta em `artifacts/`
---

### 4. `trainning_and_testing.ipynb`
- Treinamento e avaliação dos modelos:
  - Modelos testados: Regressão Linear, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost
  - Métrica principal: **Quantile Loss (Q=0.8)** para penalizar **subestimativas**
- Modelo Campeão: **CatBoost**
  - Obteve melhor Q-Loss no teste
  - Regularizado com parâmetros adequados 
  - `Optuna` para otimizaçao de hiperparâmetros (Bayesiana)
  - 10-fold CV + Early Stopping
  - Artefatos:
    - `artifacts/modelo_final_catboost.pkl`
    - `artifacts/best_catboost_params.json`

- Resultados no conjunto de teste:
```text
quantile_loss ≈ 434.73
mae           ≈ 889.77
mape          ≈ 40.10%
bias          ≈ +33.83
% subestimado ≈ 24.41%
```

### Gráficos gerados em `plots/`:
- Real vs Predito
- Boxplot: QLoss, MAE, MAPE
- Distribuição do Erro Absoluto
- Feature Importance (CatBoost)

Pelos gráficos, concluimos que:

- Real vs Predito: 
  - A maioria dos pontos segue bem a linha vermelha (ideal).
  - Alguns pontos destoam bastante, especialmente:
    - Para valores reais entre R$ 3000 e R$ 8000: o modelo subestimou ou superestimou significativamente.
  - Alguns outliers extremos (acima de R$ 10.000) foram previsões bem próximas do ideal, o que é ótimo.
- Boxplots:
  - Distribuição concentrada nas métricas (sobretudo Q-Loss e MAPE)
  - Outliers influenciam bastante o MAE — o que é esperado em regressões com caudas longas (gastos de saúde)
  - MAPE muito pequeno para maioria dos casos → a mediana está visivelmente baixa

- Histogram do erro absoluto:
  - A grande maioria dos erros absolutos está abaixo de R$ 1000
  - Poucos casos têm erro muito alto (R$ 3000+), mas eles existem — e explicam o MAE de ~R$ 890

Conclusão: desempenho muito bom na maior parte dos dados, com poucos erros extremos.

---

## Next Steps

Mesmo com um modelo robusto e regularizado, os resultados atuais (ex: MAE ~ R$ 890, MAPE ~ 40%) indicam **espaço para melhorias**. Abaixo estão sugestões de próximos passos:

### 1. Criação de Novas Variáveis
- **Variáveis combinadas** (interações entre features):
  - Ex: `idade * imc`, `classe * filhos`, `fumante * classe`
- **Transformações não lineares**:
  - Logaritmos, quadrados, raízes de variáveis com distribuição assimétrica
- **Engenharia de percentis**:
  - Criação de flags para clientes em faixas extremas de valor (`top 10%`, `bottom 10%` etc.)

### 2. Categorização de Variáveis Numéricas via Árvores de Decisão
- Aplicar a função `categorize_with_decision_tree` para discretizar variáveis numéricas de forma supervisionada com base na variável resposta.
- Essa técnica:
  - **Agrupa valores com comportamentos semelhantes** em relação ao alvo
  - **Melhora interpretabilidade**
  - Pode ser usada **junto com a variável original**

### 3. Análise de Outliers
- Erros muito altos em alguns pontos indicam **valores extremos** impactando a performance.
- Ações possíveis:
  - Remoção ou transformação de outliers
  - Criação de flags de “clientes atípicos”

### 4. Estratificação por Segmento
- Criar modelos separados por grupo (`sexo`, `classe`, `faixa_etária`, etc.)
- Modelos mais simples por cluster podem ser mais eficazes do que um modelo único complexo.

### 5. Outras Estratégias de Modelagem
- Testar regressão quantílica nativa (`quantile regression`) com LightGBM ou CatBoost
- Testar modelos com regularização robusta:
  - `ElasticNet`, `LassoLars`, etc.
- Empilhar modelos (stacking) para aproveitar diferentes pontos fortes

### 6. Expansão do Dataset
- Buscar novos dados comportamentais, histórico de compras ou geográficos
- Mais variabilidade pode ajudar modelos a generalizarem melhor
---

## Conclusão

Mesmo com apenas 3 variáveis selecionadas, o modelo demonstrou:
- Boa capacidade preditiva para os dados disponíveis
- Tendência controlada de **superestimar**, como desejado
- Erro absoluto médio relativamente alto (MAE ~ R$ 890), sugerindo espaço para melhoria com mais dados ou novas variáveis

A métrica **Q-Loss com `q=0.8`** se mostrou adequada ao objetivo de negócio: **evitar subestimar** o gasto de clientes, garantindo margem de segurança financeira.

---