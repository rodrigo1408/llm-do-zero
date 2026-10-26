# Arquitetura e matemática

Este documento acompanha o código passo a passo. Notação: `B` = tamanho do lote,
`T` = `block_size` (caracteres de contexto), `E` = `embed_dim`, `H` = `hidden_dim`,
`V` = tamanho do vocabulário.

## 1. Tokenização — `tokenizer.py`

Cada caractere distinto do corpus vira um inteiro. O id `0` é o token de fronteira `.`,
que nunca aparece no corpus (o `fit` recusa um corpus que o contenha). Vocabulário em ordem
determinística: `.` seguido dos caracteres ordenados. Caractere desconhecido em `encode` é
erro explícito (`TokenizerError`), nunca substituição silenciosa.

## 2. Dados — `data.py`

* **Divisão por sequência**, não por janela: um nome inteiro fica em um único conjunto
  (80% treino / 10% validação / 10% teste, semente fixa). Dividir por janela deixaria
  pedaços do mesmo nome em treino e validação e inflaria a métrica.
* **Janelas de contexto**: para cada posição, o contexto são os `T` ids anteriores
  (preenchidos com `0` no início) e o alvo é o próximo id; o último alvo de cada nome é `0`
  (o modelo aprende a terminar).

## 3. Modelo — `model.py`, `layers.py`

```
ids (B,T) ─► C[ids] (B,T,E) ─► reshape (B,T·E) ─► ·W1 + b1 ─► tanh ─► h (B,H) ─► ·W2 + b2 ─► logits (B,V)
```

| Camada | Forward | Backward (dado `dout = ∂L/∂saída`) |
|---|---|---|
| `Embedding` | `out = C[ids]` | `∂L/∂C[i] += Σ dout` sobre as posições com id `i` (`np.add.at`, porque ids se repetem) |
| `Linear` | `y = xW + b` | `∂L/∂W = xᵀ·dout`, `∂L/∂b = Σ_lote dout`, `∂L/∂x = dout·Wᵀ` |
| `Tanh` | `y = tanh(x)` | `∂L/∂x = dout ⊙ (1 − y²)` |

**Inicialização**: `W1 ~ N(0,1)·(5/3)/√(T·E)` (Kaiming com ganho de tanh) mantém a
pré-ativação numa faixa em que tanh não satura; `W2` é multiplicado por `0.01` para que os
logits iniciais fiquem perto de zero e a previsão inicial perto da uniforme. Resultado
observável: a perda no passo 0 é ≈ `ln(V)` (há teste para isso).

## 4. Perda — `losses.py`

Entropia cruzada média sobre o lote: `L = −(1/B) Σ log softmax(logits)[alvo]`.

`log_softmax` subtrai o máximo de cada linha antes de `exp` (estável para logits enormes).
O gradiente combinado softmax + entropia cruzada tem forma fechada:

```
∂L/∂logits = (softmax(logits) − one_hot(alvo)) / B
```

## 5. Backpropagation — `MLPLanguageModel.backward`

A regra da cadeia é aplicada na ordem inversa do forward: `head → tanh → hidden → embedding`.
Cada camada **acumula** gradientes (`+=`), por isso `zero_grad()` é chamado antes de cada passo
(`train_step_grads`).

Verificação: `tests/test_gradients.py` compara cada gradiente analítico com a diferença finita
central `(L(p+h) − L(p−h)) / 2h` (`h = 1e-6`, float64) para todas as camadas, para a perda e para
o modelo completo. Tolerância de erro relativo: `1e-6` por camada, `1e-5` no modelo inteiro.

## 6. Otimização — `optim.py`

* **SGD**: `p ← p − lr·g`.
* **AdamW**: médias móveis do gradiente (`m`) e do seu quadrado (`v`) com correção de viés,
  `p ← p − lr·m̂/(√v̂ + ε)`, e weight decay desacoplado `p ← p − lr·λ·p`.
  O decaimento é aplicado a todos os parâmetros (inclusive vieses e embeddings), por
  simplicidade — ver limitações no README.

## 7. Treinamento — `train.py`

Mini-lotes amostrados com reposição a partir de um gerador com semente própria (separado do
gerador da inicialização). A cada `eval_every` passos a perda é calculada sobre **todas** as
janelas de treino e de validação (não uma amostra). O passo 0 é registrado antes de qualquer
atualização.

**Divergência**: perda de lote não finita, ou acima de `10·ln(V)` (dez vezes pior que chutar
uniformemente), interrompe o treino e marca `diverged=True`; nesse caso as perdas "finais"
são reportadas como ausentes, nunca como o último valor antes da explosão.

## 8. Avaliação e geração — `train.evaluate`, `generate.py`, `bigram.py`

* Métrica: perda média em nats por token (perplexidade = `exp(perda)`).
* Referências: modelo uniforme (`ln V`) e bigrama por contagem com suavização de Laplace.
* Geração autorregressiva: começa com `T` tokens de fronteira, amostra o próximo caractere de
  `softmax(logits / temperatura)` e para no token de fronteira (ou em 40 caracteres).
* **Novidade**: das amostras geradas, quantas copiam um nome de treino, quantas acertam um nome
  real que o modelo nunca viu (validação) e quantas são inéditas.

## 9. Persistência

`model.save` grava os pesos e a configuração em `.npz`; `load` usa `allow_pickle=False`
(carregar um checkpoint não executa código). O tokenizador é salvo em JSON ao lado.
