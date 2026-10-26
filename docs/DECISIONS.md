# Decisões técnicas

Registro curto das escolhas que moldam o projeto e do que foi deliberadamente deixado de fora.

## D1. NumPy puro, sem autograd

**Decisão:** implementar forward e backward à mão, com NumPy como única dependência de execução.
**Por quê:** o objetivo é mostrar *como* um modelo de linguagem aprende. Com PyTorch, o
`loss.backward()` esconderia exatamente a parte que o projeto quer tornar visível.
**Custo:** cada nova camada exige derivar e testar o gradiente. É por isso que
`tests/test_gradients.py` existe: nenhuma camada entra sem verificação por diferenças finitas.

## D2. Começar por um MLP, não por um Transformer

**Decisão:** a v0.1 é o modelo de Bengio et al. (2003): embeddings + uma camada oculta.
**Por quê:** já contém todos os fundamentos (tokenização, embeddings, perda, backprop, otimização,
overfitting, geração) num modelo pequeno o bastante para ser lido inteiro em minutos. Atenção
só entra quando houver uma base testada sobre a qual comparar (ver roadmap no README).

## D3. Nível de caractere e corpus de municípios do IBGE

**Decisão:** tokenização por caractere sobre os 5.298 nomes distintos de municípios brasileiros.
**Por quê:** vocabulário pequeno (42 tokens), dados públicos e sem informação pessoal, treino
completo em segundos numa CPU, e amostras fáceis de julgar a olho ("são miguel dos vies" soa
plausível; "nto do" não). O corpus fica versionado; `scripts/fetch_ibge.py` documenta a coleta.

## D4. Divisão por sequência e conjunto de teste usado uma vez

**Decisão:** treino/validação/teste por nome inteiro; o teste só é avaliado para a configuração de
referência, fixada antes dos experimentos.
**Por quê:** comparar variações pela validação e reportar o teste de todas transformaria o teste em
mais uma validação.

## D5. Resultados gerados, não digitados

**Decisão:** `results/*.json`, `results/SUMMARY.md` e `docs/img/*.svg` são saídas de
`python -m llmzero experiments`; nenhum número é editado à mão.
**Por quê:** a documentação não pode divergir do que o código produz. Quem quiser conferir roda o
comando e compara.

## D6. Gráficos em SVG escrito à mão

**Decisão:** um gerador de SVG de ~100 linhas em vez de matplotlib.
**Por quê:** evita uma dependência pesada só para desenhar curvas, produz arquivos de poucos KB que o
GitHub renderiza e segue a identidade visual do portfólio.

## D7. Divergência é um resultado, não uma exceção

**Decisão:** uma perda não finita ou maior que `10·ln(V)` interrompe o treino e marca a execução como
divergente; as métricas finais ficam ausentes.
**Por quê:** uma taxa de aprendizado alta demais é um experimento legítimo. O resultado honesto é
"divergiu no passo N", não um número de perda do último ponto antes da explosão.

## D8. Checkpoints sem pickle

**Decisão:** `.npz` com `allow_pickle=False` + tokenizador em JSON.
**Por quê:** carregar um checkpoint de terceiros não deve poder executar código.
