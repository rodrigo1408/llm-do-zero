# llm-do-zero

**Um modelo de linguagem construído do zero em NumPy — tokenização, embeddings, backpropagation
escrita à mão, treinamento e geração de texto — sem pesos pré-treinados e sem autograd.**

[English version](README.en.md)

O modelo aprende, caractere por caractere, a "inventar" nomes de municípios brasileiros a partir dos
5.298 nomes oficiais do IBGE. É um laboratório educacional: pequeno o bastante para ser lido inteiro,
completo o bastante para mostrar cada etapa de como um modelo de linguagem aprende.

```text
$ llmzero train --out runs/default        # ~11 s em CPU
$ llmzero generate --run runs/default -n 6 --seed 3 --temperature 0.8
ampo miguel dos do matalinho
nossia
coronel mosal
iguaçu
guaporácia
timpo cochos
```

Saída real do modelo de referência. Das 200 amostras avaliadas nos experimentos (temperatura 1.0),
97% não existem no conjunto de treino.

## Situação

| Etapa | Estado | Onde |
|---|---|---|
| Tokenização por caractere | implementada e testada | [`tokenizer.py`](src/llmzero/tokenizer.py) |
| Preparação dos dados (divisão por sequência, janelas) | implementada e testada | [`data.py`](src/llmzero/data.py) |
| Embeddings, camada linear, tanh (forward + backward manuais) | implementada; gradientes verificados por diferenças finitas | [`layers.py`](src/llmzero/layers.py) |
| Entropia cruzada estável | implementada e verificada | [`losses.py`](src/llmzero/losses.py) |
| Backpropagation do modelo completo | implementada e verificada | [`model.py`](src/llmzero/model.py) |
| SGD e AdamW | implementados e testados | [`optim.py`](src/llmzero/optim.py) |
| Treino, avaliação, detecção de divergência | implementados e testados | [`train.py`](src/llmzero/train.py) |
| Geração com temperatura + métrica de novidade | implementadas e testadas | [`generate.py`](src/llmzero/generate.py) |
| Experimentos reproduzíveis | 5 experimentos, resultados versionados | [`results/SUMMARY.md`](results/SUMMARY.md) |
| **Transformer (atenção)** | **não implementado** — próxima etapa | [Roadmap](#roadmap) |

## Resultados

Perda em nats por token (menor é melhor) — números de [`results/SUMMARY.md`](results/SUMMARY.md),
gerados por `llmzero experiments`:

| Modelo | Validação | Teste |
|---|---:|---:|
| Uniforme (chute: `ln 42`) | 3.7377 | — |
| Bigrama por contagem (sem rede neural) | 2.3851 | — |
| **MLP, 18.746 parâmetros, contexto de 4 caracteres** | **1.8414** | **1.8190** |

O teste é avaliado uma única vez, para a configuração fixada antes dos experimentos.

<p align="center"><img src="docs/img/overfitting.svg" alt="Curvas de perda mostrando overfitting com 300 nomes de treino" width="720"></p>

Com só 300 nomes de treino, a perda de treino cai para ~0,8 enquanto a de validação piora depois do
passo 500: o modelo decora em vez de generalizar (24% das amostras são cópias do treino). Weight decay
reduz a perda final de validação de 3,19 para 2,61, mas o melhor ponto continua sendo parar cedo.
Os demais experimentos (taxa de aprendizado, tamanho do contexto, SGD vs. AdamW) estão em
[`results/SUMMARY.md`](results/SUMMARY.md).

## Como funciona

```text
"rio"  ─► tokenização ─► janelas  [. . . .]→r  [. . . r]→i  [. . r i]→o  [. r i o]→.
                                           │
          ids (B,4) ─► Embedding (42×12) ─► concatena (B,48) ─► Linear+tanh (B,200) ─► Linear (B,42)
                                                                                         │
                         softmax + entropia cruzada ◄────────────────────────────────────┘
                                 │
                         backward (regra da cadeia, à mão) ─► AdamW ─► próximo passo
```

A matemática de cada camada, a inicialização e as derivadas estão em
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). As escolhas de projeto e o que ficou de fora, em
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Como executar

Requisitos: Python 3.11+ e NumPy. Nenhuma GPU, chave de API ou serviço externo.

```bash
git clone https://github.com/rodrigo1408/llm-do-zero.git
cd llm-do-zero
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                                    # 37 testes, ~5 s
llmzero -v train --out runs/default       # treina o modelo de referência (~10 s em CPU)
llmzero generate --run runs/default -n 10 --temperature 0.8
llmzero experiments                       # refaz os 5 experimentos (~2 min) e regrava results/ e docs/img/
```

Todos os hiperparâmetros de `TrainConfig` viram opções: `llmzero train --block-size 8 --hidden-dim 300
--lr 0.001 --steps 10000`. `llmzero train --help` lista todas.

## Testes

| Arquivo | O que garante |
|---|---|
| [`test_gradients.py`](tests/test_gradients.py) | todo backward bate com diferenças finitas centrais (erro relativo < 1e-6), inclusive ids repetidos no embedding; softmax estável com logits de ±1000 |
| [`test_training.py`](tests/test_training.py) | perda inicial ≈ `ln(V)`; o modelo memoriza um corpus minúsculo; mesma semente ⇒ pesos idênticos; taxa absurda ⇒ divergência reportada; otimizadores minimizam uma quadrática; checkpoint ida e volta |
| [`test_tokenizer_data.py`](tests/test_tokenizer_data.py) | ida e volta com acentos; janelas iguais ao exemplo calculado à mão; divisão determinística e sem vazamento entre treino e validação |
| [`test_cli_experiments.py`](tests/test_cli_experiments.py) | treino + geração pela linha de comando; erro limpo sem checkpoint; experimento rápido gera JSON, SVG válido e resumo; nenhuma métrica inventada para execução divergente |

A CI (GitHub Actions) roda lint, testes e um treino curto + geração em Python 3.11 e 3.13.

## Estrutura

```text
src/llmzero/      tokenizer, data, layers, losses, model, optim, train, generate, bigram, plot, experiments, cli
tests/            37 testes (pytest)
data/             municipios.txt (IBGE) + SOURCE.md
results/          JSON por experimento + SUMMARY.md (gerados)
docs/             ARCHITECTURE.md, DECISIONS.md, img/*.svg (gerados)
scripts/          fetch_ibge.py (recria o corpus a partir da API do IBGE)
```

## Limitações conhecidas

- **Não é um Transformer.** O contexto é fixo (4 caracteres no modelo de referência) e cada posição
  tem pesos próprios; não há atenção.
- **Escala de brinquedo:** 18,7 mil parâmetros, 42 tokens de vocabulário, um corpus de ~62 mil
  caracteres. Os números não se comparam a modelos reais; servem para comparar variações entre si.
- **Reprodutibilidade:** resultados idênticos entre execuções na mesma máquina (verificado). Em outra
  CPU/BLAS, os últimos dígitos podem variar.
- **Uma semente por configuração:** as diferenças pequenas entre experimentos (p. ex. contexto 4 vs. 8)
  não têm intervalo de confiança.
- O weight decay é aplicado a todos os parâmetros, inclusive vieses e embeddings.
- O caminho padrão do corpus pressupõe a instalação em modo editável (`pip install -e .`); fora disso,
  use `--corpus`.

## Roadmap

1. **Atenção de uma cabeça** com backward manual e verificação de gradiente, comparada ao MLP no mesmo
   corpus e com a mesma divisão.
2. Bloco Transformer completo (atenção multi-cabeça, LayerNorm, conexões residuais, MLP).
3. Tokenização BPE treinada do zero, para um corpus de texto corrido.
4. Várias sementes por configuração, com média e desvio.

## Dados e licença

Corpus: nomes dos municípios brasileiros, API de Localidades do IBGE (dados públicos) — ver
[`data/SOURCE.md`](data/SOURCE.md). Código sob [licença MIT](LICENSE).
